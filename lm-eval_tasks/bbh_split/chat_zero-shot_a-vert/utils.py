import re

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
        wrong_group_text = a_vert.processing.refusal_candidate_group_construction()
              
        

    if task == "navigate":
        # "do you return to the starting point?"
        if refs=="yes":
            correct_group_text.append("yes, you do return to the starting point")
            wrong_group_text.append("no, you don't return to the starting point")
        else:
            correct_group_text.append("no, you don't return to the starting point")
            wrong_group_text.append("yes, you do return to the starting point")

        

    return correct_group_text, wrong_group_text
