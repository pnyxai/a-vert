import os
from functools import partial

##############################################################
############ Custom utilities for ExploreToM #################
##############################################################
# add to sys.path the current directory so that we can import postprocess_exploretom.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import a_vert
from a_vert.logger import get_logger

# Default instruction map
default_instruction = {
    "default": "Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.",
}

logger = get_logger(__name__)
# Setup A-VERT configuration from environment variables
AVERT_CONFIG = a_vert.setup(instruction_map=default_instruction)

# For backward compatibility, extract individual values
ENHANCE = AVERT_CONFIG.enhance

# ### Base ###
def base_format(example: dict, infilled_story: bool=True) -> str:
    question = example["cleaned_question"]
    prompt = ""
    story = example["infilled_story"] if infilled_story else example["story_structure"]
    prompt += story
    prompt += "\n\nQUESTION:\n"
    prompt += question
    return prompt

def format_example(example, infilled_story: bool=True):
    prompt = base_format(example, infilled_story)
    return prompt

doc_to_text_infilled_story = partial(format_example, infilled_story=True)
doc_to_text_story_structure = partial(format_example, infilled_story=False)

def doc_eval(pred, doc, task):
    """This function takes a model generated response ("pred") and the document, and evaluates the response using both exact match and A-VERT metrics.
     It returns a dictionary containing the results for both metrics.
    """
    #logger.debug("- Evaluating document", doc_id = doc.get("doc_id"))
    # ----------------------- A-VERT -------------------------------------------
    none_answer_placeholder = os.environ.get("LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER")
    if len(pred.strip()) == 0 or pred == none_answer_placeholder:
        # This is not a valid generation
        a_vert_match = False
        a_vert_correct_score = 0.0
        a_vert_wrong_score = 1.0
    else:
        correct_group_text =  doc["expected_answers"]
        wrong_group_text = doc["wrong_answers"]
        #logger.debug("- Built candidate groups", correct_group_text=correct_group_text, wrong_group_text=wrong_group_text, doc_id = doc.get("doc_id"))
        # Construct the wrong candidates group
        group_texts_dict = a_vert.processing.construct_candidate_groups(correct_group_text, 
                                wrong_group_text, 
                                ["correct", "wrong"], 
                                enhance=False,
                                )
        #logger.debug("- Constructed candidate groups", group_texts_dict=group_texts_dict, doc_id = doc.get("doc_id"))
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

        a_vert_correct_score = response_group_distribution["correct"]
        a_vert_wrong_score = response_group_distribution["wrong"]

    # --------------------------------------------------------------------------

    # Compile and return
    results = {
        "a-vert_correct_score": a_vert_correct_score, 
        "a-vert_wrong_score": a_vert_wrong_score,
        "a-vert_match": a_vert_match,
    }

    return results


def process_results(doc, results):
    """Custom processing function used to implement "a-vert" metric.
    """
    assert len(results) == 1, "only single predictions are supported"
    response = results[0]

    task = doc.get("task", "default")
    # Evaluate the document with the given model response
    result_dict = doc_eval(response, doc, task)

    return result_dict