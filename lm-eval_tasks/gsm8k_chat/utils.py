import re
import numpy as np

import a_vert


# Setup A-VERT configuration from environment variables
AVERT_SETUP = a_vert.setup()

# Extract configuration values
AVERT_METHOD = AVERT_SETUP["AVERT_METHOD"]
DOCUMENT_TEMPLATE = AVERT_SETUP["DOCUMENT_TEMPLATE"]
QUERY_TEMPLATE = AVERT_SETUP["QUERY_TEMPLATE"]
GROUPING = AVERT_SETUP["GROUPING"]
ENHANCE = AVERT_SETUP["ENHANCE"]

AVERT_MODEL_ENDPOINT = AVERT_SETUP["AVERT_MODEL_ENDPOINT"]
AVERT_ENDPOINT_TYPE = AVERT_SETUP["AVERT_ENDPOINT_TYPE"]
AVERT_MODEL_NAME = AVERT_SETUP["AVERT_MODEL_NAME"]


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



def doc_eval(pred, refs, question):
    """This function takes a model generated response ("pred") and the target
    reference ("refs") and computes the following metrics:
    - `exact_match` : A hard match between the generated string and the target
                    string.
    - `a-vert_match` : A metric that is "1" when the a-vert score of the 
                    "correct" target candidate group is higher than the "wrong" 
                    group.
    """

    # ----------------------- EXACT MATCH --------------------------------------
    # Filter response
    filtered_pred = filter_response(pred)

    # Get match
    exact_match = True
    if filtered_pred != refs:
        exact_match = False

    # ----------------------- A-VERT -------------------------------------------
    # Generate other numbers
    correct_group_text, wrong_group_text = get_gsm8k_options(refs, question)
    # Construct the wrong candidates group
    group_texts_dict = a_vert.processing.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=ENHANCE,
                               )

    # Process all candidate groups
    response_group_distribution, _ = a_vert.processing.get_candidate_groups_embedings_ranking(pred,
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
    # implementation
    assert len(results) == 1, "only single predictions are supported"

    
    # Get the data
    response = results[0]
    target = doc["answer"]
    question = doc["question"]

    # Evaluate the document with the given model response
    results = doc_eval(response, target, question)

    return results


# ------------------------------------------------------------------------------
# --------------------- gsm8k specific code ------------------------------------
# ------------------------------------------------------------------------------

def get_gsm8k_options(question_target, question):

    # Get target number
    target_num = int(question_target.split("#### ")[-1])
    # Set other numbers
    other_options = [
        np.floor(target_num*0.1),
        np.floor(target_num*0.5),
        np.ceil(target_num*1.25),
        np.ceil(target_num*1.8),
    ]
    other_options = np.unique(other_options)
    other_options = [int(a) for a in other_options if a != target_num]

    wrong_group_text = [f"{a}" for a in other_options]
    correct_group_text = [f"{target_num}"]

    return correct_group_text, wrong_group_text
