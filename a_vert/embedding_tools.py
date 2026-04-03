import numpy as np
import requests
import json
from scipy import spatial

from a_vert.logger import get_logger

logger = get_logger(__name__)

_RETRY_EXCEPTIONS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


def _post_with_retry(url, payload, timeout=20, max_retries=3, **log_context):
    """POST to `url` with timeout and retry on transient network errors.

    Returns the raw `requests.Response` on HTTP 200. Raises `ValueError` after
    exhausting retries (on Timeout/ConnectionError) or immediately when the
    server replies with a non-200 status code.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    headers = {"Content-Type": "application/json"}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url, data=json.dumps(payload), headers=headers, timeout=timeout
            )
            break
        except _RETRY_EXCEPTIONS as exc:
            last_exc = exc
            logger.warning(
                "Endpoint call failed, retrying",
                attempt=attempt,
                max_retries=max_retries,
                url=url,
                error=str(exc),
                **log_context,
            )
    else:
        logger.error(
            "Endpoint call failed after all retries",
            max_retries=max_retries,
            url=url,
            **log_context,
        )
        raise ValueError("Failed to call endpoint after retries.") from last_exc

    if response.status_code != 200:
        logger.error(
            "Endpoint returned non-200 status",
            status_code=response.status_code,
            response_body=response.text,
            url=url,
            **log_context,
        )
        raise ValueError("Failed to call endpoint.")

    return response


def tei_embedding_call(text, tei_endpoint, timeout=20, max_retries=3):
    """Calls the Text-Embedding-Inference endpoint and return the embeddings
    array.
    """
    payload = {"inputs": text, "truncate": True, "truncation_direction": "Left"}
    response = _post_with_retry(
        tei_endpoint + "/embed", payload, timeout=timeout, max_retries=max_retries
    )
    return np.array(json.loads(response.text))


def vllm_embedding_call(
    text, vllm_endpoint, vllm_model_name, max_len=-1, timeout=20, max_retries=3
):
    """Calls the vLLM endpoint and return the embeddings array."""
    payload = {
        "input": text,
        "model": vllm_model_name,
        "encoding_format": "float",
        "truncate_prompt_tokens": max_len,
    }
    response = _post_with_retry(
        vllm_endpoint + "/v1/embeddings",
        payload,
        timeout=timeout,
        max_retries=max_retries,
        model=vllm_model_name,
    )
    data = json.loads(response.text)
    return np.array([a["embedding"] for a in data["data"]])


def get_embedding(text, endpoint, endpoint_type, model_name=None, max_batch_size=32):
    """Call the Text-Embedding-Inference endpoint handling the endpoint batch
    size.
    """
    # Assign endpoint call
    if endpoint_type == "tei":

        def embedding_call(x):
            return tei_embedding_call(x, endpoint)
    elif endpoint_type == "vllm" or endpoint_type == "openai":
        if model_name is None:
            raise ValueError("Model name is required for vllm/openai endpoint.")

        def embedding_call(x):
            return vllm_embedding_call(x, endpoint, model_name)
    else:
        raise ValueError("Endpoint type not supported")

    # Calculate embeddings for the text list
    if isinstance(text, list):
        if len(text) > max_batch_size:
            partial = embedding_call(text[:max_batch_size])
            last = max_batch_size
            while last < len(text):
                if last + max_batch_size > len(text):
                    part = embedding_call(text[last:])
                else:
                    part = embedding_call(text[last : last + max_batch_size])
                last += max_batch_size
                partial = np.concatenate([partial, part], axis=0)
            return partial

        else:
            return embedding_call(text)
    else:
        return embedding_call(text)


def check_and_apply_template(template, placeholder, text):
    """
    Apply a template to a text, replacing a placeholder.
    If the template is None, returns the original text.
    """
    if template is not None:
        # Check if the template contains the string to replace
        if placeholder not in template:
            err = f"Template must contain a {placeholder} placeholder. Template: '{template}'"
            raise ValueError(err)
        # Replace
        prompt = template.replace(placeholder, text)
    else:
        prompt = text

    return prompt


def calculate_embedding_distances(
    model_response,
    batch,
    endpoint,
    endpoint_type,
    model_name=None,
    query_template=None,
    document_template=None,
    distance_fn=spatial.distance.cosine,
    batch_size=32,
):
    # Calculate targets embeddings
    batch_to_embedding = [
        check_and_apply_template(document_template, "{document}", t) for t in batch
    ]
    targets_embeddings = get_embedding(
        batch_to_embedding,
        endpoint,
        endpoint_type,
        model_name=model_name,
        max_batch_size=batch_size,
    )
    # Get model response embedding
    model_response_to_embedding = check_and_apply_template(
        query_template, "{query}", model_response
    )
    model_response_embedding = np.squeeze(
        get_embedding(
            model_response_to_embedding,
            endpoint,
            endpoint_type,
            model_name=model_name,
            max_batch_size=batch_size,
        )
    )

    # Calculate the distances
    all_distances = [
        1 - distance_fn(model_response_embedding, this_emb)
        for this_emb in targets_embeddings
    ]

    return all_distances


def tei_rerank_call(query, targets, tei_rerank_endpoint, timeout=20, max_retries=3):
    """Calls the TEI endpoint and return the ranking scores as an array, in the
    same order as they were provided.
    """
    payload = {
        "query": query,
        "texts": targets,
        "truncate": True,
        "truncation_direction": "Left",
    }
    response = _post_with_retry(
        tei_rerank_endpoint + "/rerank",
        payload,
        timeout=timeout,
        max_retries=max_retries,
    )
    response = json.loads(response.text)

    total_ranks = len(response)
    if total_ranks != len(targets):
        logger.error(
            "Mismatch between response and target count",
            query=query,
            num_targets=len(targets),
            num_responses=total_ranks,
            targets=targets,
            response=response,
        )
        raise ValueError(
            "Received less scores than targets from endpoint, cannot continue."
        )

    # Process results and get scores in same order as targets
    all_scores = np.zeros((total_ranks))
    for ranked in response:
        all_scores[ranked["index"]] = ranked["score"]

    return all_scores


def vllm_rerank_call(
    query,
    targets,
    vllm_rerank_endpoint,
    vllm_model_name,
    max_len=-1,
    timeout=20,
    max_retries=3,
):
    """Calls the vLLM endpoint and return the ranking scores as an array, in the
    same order as they were provided.
    """
    payload = {
        "query": query,
        "documents": targets,
        "model": vllm_model_name,
        "truncate_prompt_tokens": max_len,
    }
    response = _post_with_retry(
        vllm_rerank_endpoint + "/v1/rerank",
        payload,
        timeout=timeout,
        max_retries=max_retries,
        model=vllm_model_name,
    )
    response = json.loads(response.text)

    total_ranks = len(response["results"])
    if total_ranks != len(targets):
        logger.error(
            "Mismatch between response and target count",
            query=query,
            num_targets=len(targets),
            num_responses=total_ranks,
            targets=targets,
            response=response,
        )
        raise ValueError(
            "Received less scores than targets from endpoint, cannot continue."
        )

    # Process results and get scores in same order as targets
    all_scores = np.zeros((total_ranks))
    for ranked in response["results"]:
        all_scores[ranked["index"]] = ranked["relevance_score"]

    return all_scores


def get_rerank(
    query, targets, endpoint, endpoint_type, model_name=None, max_batch_size=32
):
    """Call the .0.. batch
    size.
    """
    # Assign endpoint call
    if endpoint_type == "tei":

        def reranking_call(x, y):
            return tei_rerank_call(x, y, endpoint)
    elif endpoint_type == "vllm" or endpoint_type == "openai":
        if model_name is None:
            raise ValueError("Model name is required for vllm/openai endpoint.")

        def reranking_call(x, y):
            return vllm_rerank_call(x, y, endpoint, model_name)
    else:
        raise ValueError("Endpoint type not supported")

    # Discount query place
    max_batch_size -= 1
    # Calculate embeddings for the text list
    if isinstance(targets, list):
        if len(targets) > max_batch_size:
            partial = reranking_call(query, targets[:max_batch_size])
            last = max_batch_size
            while last < len(targets):
                if last + max_batch_size > len(targets):
                    part = reranking_call(query, targets[last:])
                else:
                    part = reranking_call(query, targets[last : last + max_batch_size])
                last += max_batch_size
                partial = np.concatenate([partial, part], axis=0)
            return partial

        else:
            return reranking_call(query, targets)
    else:
        return reranking_call(query, targets)


def calculate_reranking_distances(
    model_response,
    batch,
    endpoint,
    endpoint_type,
    model_name=None,
    query_template=None,
    document_template=None,
    batch_size=32,
):
    # Calculate targets embeddings
    batch_to_rank = [
        check_and_apply_template(document_template, "{document}", t) for t in batch
    ]
    model_response_to_rank = check_and_apply_template(
        query_template, "{query}", model_response
    )

    # Get model response rerank
    all_scores = get_rerank(
        model_response_to_rank,
        batch_to_rank,
        endpoint,
        endpoint_type,
        model_name=model_name,
        max_batch_size=batch_size,
    )

    return all_scores
