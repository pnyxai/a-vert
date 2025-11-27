import random
import re
import datasets

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



def doc_eval(pred, refs, question, choices):
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
    # Generate options groups
    correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs  = get_gpqa_options(refs, question, choices)

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

   
    # Get the data

    response = results[0]
    target = preprocess(doc["Correct Answer"])
    question = doc["Question"]
    choices = doc["choices"]


    # Evaluate the document with the given model response
    results = doc_eval(response, target, question, choices)

    return results



# ------------------------------------------------------------------------------
# --------------------- GPQA specific code -------------------------------------
# ------------------------------------------------------------------------------


def get_gpqa_options(question_target, question, choices):


    correct_group_text = list()
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx in range(len(choices)):
        if choices[idx] == question_target:
            if len(correct_group_idxs) == 0:
                correct_group_text.append(choices[idx])
                correct_group_idxs.append(idx)
            else:
                print(f"WARNING: Duplicated target found.\n\t{choices[idx]}\n\t{question_target}")
        else:
            wrong_group_text.append(choices[idx])
            wrong_group_idxs.append(idx)
            
    if len(wrong_group_text) == 0:
        print(f"wrong group text is empty! patching with refusals and continuing...\n\t{question_target}\n\t{choices}")
        wrong_group_text = a_vert.processing.refusal_candidate_group_construction()
        for idx in range(len(wrong_group_text)):
            wrong_group_idxs.append(idx+1)
    
    
    assert len(correct_group_text) == len(correct_group_idxs)
    assert len(wrong_group_idxs) == len(wrong_group_text)

    return correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs 




def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)







