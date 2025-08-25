import re
import numpy as np
import os
from scipy import spatial

from a_vert import processing as a_vert
from a_vert import embedding_tools as a_vert_tools

# ---- Different a-vert configs
# 
# Qwen3-Reranker Family : Qwen3-Reranker-0.6B-seq-cls, Qwen3-Reranker-4B-seq-cls
#     
# AVERT_METHOD = "rerank"
# DOCUMENT_TEMPLATE = "<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# QUERY_TEMPLATE = """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n <Instruct>: Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.\n<Query>: {query}\n"""
# 
# Rerankers without instruction : gte-reranker-modernbert-base, jina-reranker-v2-base-multilingual
# 
AVERT_METHOD = "rerank"
DOCUMENT_TEMPLATE = None
QUERY_TEMPLATE = None
# 
# Embedding with instruction : Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, multilingual-e5-large-instruct 
# 
# AVERT_METHOD = "embedding"
# DOCUMENT_TEMPLATE = None
# QUERY_TEMPLATE = 'Instruct: Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.\nQuery:{query}'
# 
# Embedding without instruction : gte-modernbert-base
# 
# AVERT_METHOD = "embedding"
# DOCUMENT_TEMPLATE = None
# QUERY_TEMPLATE = None

GROUPING="max"

ENCHANCE = True

MAX_LEN = 1000

# This environment variable contains the endpoint to the selected model
AVERT_MODEL_ENDPOINT = os.getenv("AVERT_MODEL_ENDPOINT", None)
if AVERT_MODEL_ENDPOINT is None:
    raise ValueError("AVERT_MODEL_ENDPOINT environment variable is not set. This is required for A-VERT to function.")
AVERT_ENDPOINT_TYPE = os.getenv("AVERT_ENDPOINT_TYPE", None)
if AVERT_ENDPOINT_TYPE is None:
    raise ValueError("AVERT_ENDPOINT_TYPE environment variable is not set. This is required for A-VERT to function.")
AVERT_MODEL_NAME = os.getenv("AVERT_MODEL_NAME", None)
if AVERT_MODEL_NAME is None and  (AVERT_ENDPOINT_TYPE == "vllm" or AVERT_ENDPOINT_TYPE=="openai"):
    raise ValueError("AVERT_MODEL_NAME environment variable is not set. This is required for vLLM or OpenAI endpoint to function.")


# Optional for SemScore
SEMSCORE_MODEL_ENDPOINT = os.getenv("SEMSCORE_MODEL_ENDPOINT", None)
if SEMSCORE_MODEL_ENDPOINT is None:
    print("SEMSCORE_MODEL_ENDPOINT environment variable is not set. This is required for SemScore to function.")
SEMSCORE_ENDPOINT_TYPE = os.getenv("SEMSCORE_ENDPOINT_TYPE", None)
if SEMSCORE_ENDPOINT_TYPE is None:
    print("SEMSCORE_ENDPOINT_TYPE environment variable is not set. This is required for SemScore to function.")
SEMSCORE_MODEL_NAME = os.getenv("SEMSCORE_MODEL_NAME", None)
if SEMSCORE_MODEL_NAME is None and  (SEMSCORE_ENDPOINT_TYPE == "vllm" or SEMSCORE_MODEL_ENDPOINT=="openai"):
    raise ValueError("SEMSCORE_MODEL_NAME environment variable is not set. This is required for vLLM or OpenAI endpoint to function.")

SEMSCORE_THRESHOLD = 0.6237373737373737

def filter_response(pred):
    """This function is used by the "exact_match" metric to try to clean the
    model generated answer.
    """

    try:
        # Filter everything after the first break line
        filtered_pred = re.findall(r"^(.*?)(?=\n|$)", pred)[0].strip()
        # Remove leading white spaces
        filtered_pred = filtered_pred.lstrip()
        # function to ignore right white spaces or line breaks
        filtered_pred = re.findall(r"^(.*?)\s*$", filtered_pred)[0].strip()
    except:
        filtered_pred = "[invalid]"

    return filtered_pred



def doc_eval(pred, target_idx, choices):
    """This function takes a model generated response ("pred") and the target
    reference ("refs") and computes the following metrics:
    - `exact_match` : A hard match between the generated string and the target
                    string.
    - `a-vert_match` : A metric that is "1" when the a-vert score of the 
                    "correct" target candidate group is higher than the "wrong" 
                    group.
    """


    correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs = get_mmlu_options(target_idx, choices)
    target = choices[target_idx]

    # ----------------------- EXACT MATCH --------------------------------------
    # Filter response
    filtered_pred = filter_response(pred)

    # Get match
    exact_match = True
    if filtered_pred != target:
        exact_match = False

    # ----------------------- A-VERT -------------------------------------------
    # Construct the wrong candidates group
    group_texts_dict = a_vert.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=ENCHANCE,
                               with_options=ENCHANCE,
                               option_symbol="letters",
                               correct_group_idxs=correct_group_idxs,
                               wrong_group_idxs=wrong_group_idxs
                               )

    # Process all candidate groups
    response_group_distribution, _ = a_vert.get_candidate_groups_embedings_ranking(pred,
                                           group_texts_dict,
                                           AVERT_MODEL_ENDPOINT,
                                           AVERT_ENDPOINT_TYPE,
                                            AVERT_METHOD,
                                           model_name=AVERT_MODEL_NAME,
                                           query_template=QUERY_TEMPLATE,
                                           document_template=DOCUMENT_TEMPLATE,
                                           grouping_method=GROUPING, 
                                           verbose=False,
                                           max_len=MAX_LEN
                                           )
    # Check if this is a match
    a_vert_match = True
    if response_group_distribution["correct"] < response_group_distribution["wrong"]:
        a_vert_match = False

    # ----------------------- SemScore -----------------------------------------
    if SEMSCORE_MODEL_ENDPOINT is not None:
        # Embed target
        target_emb = np.squeeze(a_vert_tools.get_embedding(target, SEMSCORE_MODEL_ENDPOINT, SEMSCORE_ENDPOINT_TYPE, model_name=SEMSCORE_MODEL_NAME))
        # Embed response
        response_emb = np.squeeze(a_vert_tools.get_embedding(pred, SEMSCORE_MODEL_ENDPOINT, SEMSCORE_ENDPOINT_TYPE, model_name=SEMSCORE_MODEL_NAME))
        # Calculate cosine similarity
        semscore = 1-spatial.distance.cosine(response_emb, target_emb)
        # Apply threshold
        semscore_match = True
        if semscore < SEMSCORE_THRESHOLD:
            semscore_match = False
    else:
        semscore = 0
        semscore_match = False

    # --------------------------------------------------------------------------

    # Compile and return
    results = {
        "exact_match": exact_match,
        "a-vert_correct_score": response_group_distribution["correct"], 
        "a-vert_wrong_score": response_group_distribution["wrong"],
        "a-vert_match": a_vert_match,
        "semscore": semscore,
        "semscore_match": semscore_match,
    }

    return results

def process_results(doc, results):
    """Custom processing function used to implement "a-vert" metric.
    """
       
    # Get the data
    response = results[0]
    target_idx = doc["answer"]
    choices = doc["choices"]

    # Evaluate the document with the given model response
    results = doc_eval(response, target_idx, choices)

    return results

# ------------------------------------------------------------------------------
# --------------------- MMLU specific code -------------------------------------
# ------------------------------------------------------------------------------

def get_mmlu_options(target_idx, choices):

    correct_group_text = list()
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx in range(len(choices)):
        if idx == target_idx:
            correct_group_text.append(choices[idx])
            correct_group_idxs.append(idx)
        else:
            wrong_group_text.append(choices[idx])
            wrong_group_idxs.append(idx)

    return correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs


