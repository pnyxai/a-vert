import numpy as np
import requests
import json
from scipy import spatial


def tei_embedding_call(text, tei_endpoint):
    """Calls the Text-Embedding-Inference endpoint and return the embeddings
    array.
    """
    payload = {"inputs": text, 
               "truncate": True, 
               "truncation_direction": "Left"
               }
    headers = {"Content-Type": "application/json"}
    response = requests.post(tei_endpoint+'/embed', data=json.dumps(payload), headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        raise ValueError("Failed to call endpoint.")
    embedding = np.array(json.loads(response.text))
    return embedding

def vllm_embedding_call(text, vllm_endpoint, vllm_model_name):
    """Calls the vLLM endpoint and return the embeddings array.
    """
    payload = {
        "input": text,
        "model": vllm_model_name,
        "encoding_format": "float"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(vllm_endpoint+'/v1/embeddings', data=json.dumps(payload), headers=headers)
    if response.status_code == 400:
        if "This model's maximum context length is" in json.loads(response.text)['error']["message"]:
            # vLLM does not crop embedding inputs... so we hack away...
            print(response.text)
            # Too long, lets crop and call   
            this_len = len(text.split(" "))
            new_len = int(this_len/2)
            print(new_len)
            text = truncate(text, new_len)
            return vllm_embedding_call(text, vllm_endpoint, vllm_model_name)

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        raise ValueError("Failed to call endpoint.")
    response = json.loads(response.text)
    embedding = np.array([ a['embedding'] for a in response['data']])
    return embedding




def get_embedding(text, endpoint, endpoint_type, model_name=None, max_batch_size=32, max_len = -1):
    """Call the Text-Embedding-Inference endpoint handling the endpoint batch 
    size.
    """
    # Assign endpoint call
    if endpoint_type == "tei":
        embedding_call = lambda x: tei_embedding_call(x, endpoint)
    elif endpoint_type == "vllm" or endpoint_type == "openai":
        if model_name is None:
            raise ValueError("Model name is required for vllm/openai endpoint.")
        embedding_call = lambda x: vllm_embedding_call(x, endpoint, model_name)
    else:
        raise ValueError("Endpoint type not supported")

    # Calculate embeddings for the text list
    if isinstance(text, list):
        if len(text) > max_batch_size:
            partial = embedding_call(text[:max_batch_size])
            last = max_batch_size
            while last < len(text):
                if last+max_batch_size > len(text):
                    part = embedding_call(text[last:])
                else:
                    part = embedding_call(text[last:last+max_batch_size])
                last += max_batch_size
                partial = np.concatenate([partial, part], axis=0)
            return partial
                
        else:
            return embedding_call(text)
    else:
        return embedding_call(text)

def truncate(prompt, max_len):
    if max_len > 0:
        if len(prompt.split(" ")) > max_len:
            prompt = " ".join(prompt.split(" ")[-max_len:])
    return prompt

def check_and_apply_template(template, placeholder, query, max_len=-1):
    if template is not None:
        # Check if the template contains the string to replace
        if placeholder not in template:
            err = f"Instruction must contain a {placeholder} placeholder"
            raise ValueError(err)
        # Replace
        prompt = template.replace(placeholder,query)
    else:
        prompt =  query
    
    return truncate(prompt, max_len)

    
def calculate_embedding_distances(model_response,
                                  batch,
                                  endpoint,
                                  endpoint_type, 
                                  model_name=None,
                                  query_template=None,
                                  document_template=None,
                                  distance_fn = spatial.distance.cosine, 
                                  batch_size=32,
                                  max_len=-1,
                                  ):
    # Calculate targets embeddings
    batch_to_embedding = [check_and_apply_template(document_template, "{document}", t, max_len=max_len) for t in batch]
    targets_embeddings = get_embedding(batch_to_embedding, endpoint, endpoint_type, model_name=model_name, max_batch_size=batch_size)
    # Get model response embedding
    model_response_to_embedding = check_and_apply_template(query_template, "{query}", model_response, max_len=max_len)
    model_response_embedding = np.squeeze(get_embedding(model_response_to_embedding, endpoint, endpoint_type, model_name=model_name, max_batch_size=batch_size))

    # Calculate the distances
    all_distances = [1 - distance_fn(model_response_embedding, this_emb) for this_emb in targets_embeddings]

    return all_distances



def tei_rerank_call(query, targets, tei_rerank_endpoint):
    """Calls the TEI endpoint and return the ranking scores as an array, in the 
    same order as they were provided.
    """
    # Get ranks
    payload = {
            "query": query,
            "texts": targets,
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(tei_rerank_endpoint+'/rerank', data=json.dumps(payload), headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        raise ValueError("Failed to call endpoint.")
    response = json.loads(response.text)

    total_ranks = len(response)
    if total_ranks != len(targets):
        print(query)
        print(targets)
        print(response)
        raise ValueError("Received less scores than targets from endpoint, cannot continue.")

    # Process results and get scores in same order as targets
    all_scores = np.zeros((total_ranks))
    for ranked in response:
        all_scores[ranked['index']] = ranked['score']

    return all_scores
    



def vllm_rerank_call(query, targets, vllm_rerank_endpoint, vllm_model_name):
    """Calls the vLLM endpoint and return the ranking scores as an array, in the 
    same order as they were provided.
    """
    # Get ranks
    payload = {
            "query": query,
            "documents": targets,
            "model": vllm_model_name,
            # "max_tokens_per_doc": 16000,
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(vllm_rerank_endpoint+'/v1/rerank', data=json.dumps(payload), headers=headers)
    if response.status_code == 400:
        if "This model's maximum context length is" in json.loads(response.text)['error']["message"]:
            # vLLM does not crop embedding inputs... so we hack away...
            print(response.text)
            # Too long, lets crop and call   
            len_max = len(query.split(" "))
            for this_target in targets:
                if len(this_target.split(" ")) > len_max:
                    len_max = len(this_target.split(" "))
            new_len = int(len_max/2)
            print(new_len)
            query = truncate(query, new_len)
            targets = [truncate(a, new_len) for a in targets]
            return vllm_rerank_call(query, targets, vllm_rerank_endpoint, vllm_model_name)
    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        raise ValueError("Failed to call endpoint.")

    response = json.loads(response.text)

    total_ranks = len(response['results'])
    if total_ranks != len(targets):
        print(query)
        print(targets)
        print(response)
        print(len(total_ranks))
        print(len(targets))
        raise ValueError("Received less scores than targets from endpoint, cannot continue.")

    # Process results and get scores in same order as targets
    all_scores = np.zeros((total_ranks))
    for ranked in response['results']:
        all_scores[ranked['index']] = ranked['relevance_score']

    return all_scores
    

def get_rerank(query, targets, endpoint, endpoint_type, model_name=None, max_batch_size=32):
    """Call the .0.. batch 
    size.
    """
    # Assign endpoint call
    if endpoint_type == "tei":
        reranking_call = lambda x,y: tei_rerank_call(x, y, endpoint)
    elif endpoint_type == "vllm" or endpoint_type == "openai":
        if model_name is None:
            raise ValueError("Model name is required for vllm/openai endpoint.")
        reranking_call = lambda x,y: vllm_rerank_call(x, y, endpoint, model_name)
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
                if last+max_batch_size > len(targets):
                    part = reranking_call(query, targets[last:])
                else:
                    part = reranking_call(query, targets[last:last+max_batch_size])
                last += max_batch_size
                partial = np.concatenate([partial, part], axis=0)
            return partial
                
        else:
            return reranking_call(query, targets)
    else:
        return reranking_call(query, targets)

def calculate_reranking_distances(model_response,
                                  batch,
                                  endpoint,
                                  endpoint_type,
                                  model_name=None,
                                  query_template=None,
                                  document_template=None,
                                  batch_size=32,
                                  max_len =-1
                                  ):
    
    # Calculate targets embeddings
    batch_to_rank = [check_and_apply_template(document_template, "{document}", t, max_len=max_len) for t in batch]
    model_response_to_rank = check_and_apply_template(query_template, "{query}", model_response, max_len=max_len)

    
    # Get model response rerank
    all_scores = get_rerank(model_response_to_rank, batch_to_rank, endpoint, endpoint_type, model_name=model_name, max_batch_size=batch_size)

    return all_scores
