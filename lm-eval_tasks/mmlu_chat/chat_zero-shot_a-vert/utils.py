import re
import a_vert


# Default instruction map
default_instruction = {
    "default": "Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.",
}

# Setup A-VERT configuration from environment variables
AVERT_CONFIG = a_vert.setup(instruction_map=default_instruction)

# For backward compatibility, extract individual values
ENHANCE = AVERT_CONFIG.enhance

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
    except Exception:
        filtered_pred = "[invalid]"

    return filtered_pred



def doc_eval(pred, target_idx, choices, task):
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
    group_texts_dict = a_vert.processing.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=ENHANCE,
                               with_options=ENHANCE,
                               option_symbol="letters",
                               correct_group_idxs=correct_group_idxs,
                               wrong_group_idxs=wrong_group_idxs
                               )

    # Process all candidate groups
    response_group_distribution, _ = a_vert.processing.get_candidate_groups_embedings_ranking(
        pred,
        group_texts_dict,
        AVERT_CONFIG,
        task=task if task else "default",
    )
    # Check if this is a match
    a_vert_match = True
    if response_group_distribution["correct"] < response_group_distribution["wrong"]:
        a_vert_match = False



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
       
    # Get the data
    response = results[0]
    target_idx = doc["answer"]
    choices = doc["choices"]
    task = doc.get("task", "default")

    # Evaluate the document with the given model response
    result_dict = doc_eval(response, target_idx, choices, task=task)

    return result_dict

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
