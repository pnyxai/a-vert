import re
import os

from a_vert import processing as a_vert

# ---- Different a-vert configs
# 
# Qwen3-Reranker Family : Qwen3-Reranker-0.6B-seq-cls, Qwen3-Reranker-4B-seq-cls
#     
AVERT_METHOD = "rerank"
DOCUMENT_TEMPLATE = "<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUERY_TEMPLATE = """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n <Instruct>: Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.\n<Query>: {query}\n"""
# 
# Rerankers without instruction : gte-reranker-modernbert-base, jina-reranker-v2-base-multilingual
# 
# AVERT_METHOD = "rerank"
# DOCUMENT_TEMPLATE = None
# QUERY_TEMPLATE = None
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



def doc_eval(pred, options, target_idx, question, task):
    """This function takes a model generated response ("pred") and the 

    """

    refs = options[target_idx]

    # ----------------------- EXACT MATCH --------------------------------------
    # Filter response
    filtered_pred = filter_response(pred)

    # Get match
    exact_match = True
    if filtered_pred != refs:
        exact_match = False

    # ----------------------- A-VERT -------------------------------------------
    # Get other elements from the bAbI world
    correct_group_text, wrong_group_text = get_bbh_options(refs, question, options, task)
    # Construct the wrong candidates group
    group_texts_dict = a_vert.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=ENCHANCE,
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
                                           )
    # Check if this is a match
    a_vert_match = True
    if response_group_distribution["correct"] < response_group_distribution["wrong"]:
        a_vert_match = False

    # --------------------------------------------------------------------------

    # Compile and return
    results = {
        "exact_match": exact_match,
        "a-vert_correct_score": response_group_distribution["correct"], 
        "a-vert_wrong_score": response_group_distribution["wrong"],
        "a-vert_match": a_vert_match,
    }



    return results

def process_results(doc, results):
    """Custom processing function used to implement "a-vert" metric.
    """

    # Assert we are evaluating a single target. This is a limitation of this 
    # bAbI implementation
    assert len(results) == 1, "only single predictions are supported"

    # Get the data
    response = results[0]
    target = doc["target_idx"]
    options = doc["options"]
    question = doc["input"]
    task = doc["task"]

    # Evaluate the document with the given model response
    results = doc_eval(response, options, target, question, task)

    return results



# ------------------------------------------------------------------------------
# --------------------- BBH specific code --------------------------------------
# ------------------------------------------------------------------------------

def get_bbh_options(refs, question, options, task):

    correct_group_text = [refs]
    wrong_group_text = [ a for a in options if a != refs]

    if len(wrong_group_text) == 0:
        print(f"wrong group text is empty! patching with refusals and continuing...\n\t{refs}\n\t{options}")
        wrong_group_text = a_vert.refusal_candidate_group_construction()
              
        

    if task == "navigate":
        # "do you return to the starting point?"
        if refs=="yes":
            correct_group_text.append("yes, you do return to the starting point")
            wrong_group_text.append("no, you don't return to the starting point")
        else:
            correct_group_text.append("no, you don't return to the starting point")
            wrong_group_text.append("yes, you do return to the starting point")

        

    return correct_group_text, wrong_group_text
