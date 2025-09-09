from functools import partial

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




# ### Base ###
def base_format(example: dict, include_options: bool) -> str:
    prompt = ""
    prompt += "## World enumeration ##\n"
    world = example["world_enumerate"]
    prompt += world
    prompt += "\n\n## Story ##\n"
    story = example["story"]
    prompt += story
    prompt += "\nQuestion: "
    question = example["question"]
    prompt += question
    if include_options:
        prompt += "Options:\n"
        options = example["options"]
        for i, opt in enumerate(options):
            prompt += "{}\n".format(opt)
    return prompt


def format_example(example, include_options: bool, including_answer: bool):
    prompt = base_format(example, include_options)
    return prompt


doc_to_text = partial(format_example, include_options=False, including_answer=False)




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



def doc_eval(pred, options, answers, question, task):
    """This function takes a model generated response ("pred") and the 

    """

    # ----------------------- EXACT MATCH --------------------------------------
    # Filter response
    filtered_pred = filter_response(pred)

    # Get match
    exact_match = False
    for answ in answers:
        if filtered_pred == answ:
            exact_match = True

    # ----------------------- A-VERT -------------------------------------------
    # Get other elements from the bAbI world
    correct_group_text, wrong_group_text = get_babisteps_options(answers, question, options, task)
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
    # print(doc)
    response = results[0]
    answer = doc["contextualized_answer"]
    options = doc["contextualized_options"]
    question = doc["question"]
    task = "" #doc["task"]

    # Evaluate the document with the given model response
    results = doc_eval(response, options, answer, question, task)

    return results



# ------------------------------------------------------------------------------
# --------------------- babisteps specific code --------------------------------
# ------------------------------------------------------------------------------

def get_babisteps_options(answers, question, options, task):

    correct_group_text = answers
    wrong_group_text = list()
    for option in options:
        add = True
        for answ in answers:
            if answ == option:
                add = False
        if add:
            wrong_group_text.append(option)       

    return correct_group_text, wrong_group_text