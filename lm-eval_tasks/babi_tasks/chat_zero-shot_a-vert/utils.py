import re
import itertools
import numpy as np
from copy import deepcopy
import os
from scipy import spatial

from a_vert import processing as a_vert
from a_vert import embedding_tools as a_vert_tools

# ---- Here we set the a-vert configuration for Qwen3-Reranker-0.6B-seq-cls
# Method : embedding / rerank
AVERT_METHOD = "rerank"
# These are the templates required by the model
DOCUMENT_TEMPLATE = "<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUERY_TEMPLATE = """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n <Instruct>: Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.\n<Query>: {query}\n"""
# Grouping to be applied to candidate groups
GROUPING="max"

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

SEMSCORE_THRESHOLD = 0.75

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



def doc_eval(pred, refs):
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
    # Get other elements from the bAbI world
    wrong_group_text = get_babi_options(refs)
    # Construct the wrong candidates group
    group_texts_dict = a_vert.construct_candidate_groups([refs], 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=True,
                               )

    # Process all candidate groups
    response_group_distribution, _ = a_vert.get_candidate_groups_embedings_ranking([pred],
                                           group_texts_dict,
                                           AVERT_MODEL_ENDPOINT,
                                           AVERT_ENDPOINT_TYPE,
                                            AVERT_METHOD,
                                           model_name=AVERT_MODEL_NAME,
                                           query_template=QUERY_TEMPLATE,
                                           document_template=DOCUMENT_TEMPLATE,
                                           grouping_method=GROUPING, 
                                           verbose=False
                                           )
    # Check if this is a match
    a_vert_match = True
    if response_group_distribution["correct"] < response_group_distribution["wrong"]:
        a_vert_match = False

    # ----------------------- SemScore -----------------------------------------
    if SEMSCORE_MODEL_ENDPOINT is not None:
        # Embed target
        target_emb = np.squeeze(a_vert_tools.get_embedding([refs], SEMSCORE_MODEL_ENDPOINT, SEMSCORE_ENDPOINT_TYPE, model_name=SEMSCORE_MODEL_NAME))
        # Embed response
        response_emb = np.squeeze(a_vert_tools.get_embedding([pred], SEMSCORE_MODEL_ENDPOINT, SEMSCORE_ENDPOINT_TYPE, model_name=SEMSCORE_MODEL_NAME))
        # Calculate cosine similarity
        semscore = spatial.distance.cosine(response_emb, target_emb)
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
        "a-vert_match": a_vert_match,
        "semscore": semscore,
        "semscore_match": semscore_match,
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
    target = doc["answer"]

    # Evaluate the document with the given model response
    results = doc_eval(response, target)

    return results



# ------------------------------------------------------------------------------
# --------------------- bAbI specific code -------------------------------------
# ------------------------------------------------------------------------------

def get_babi_options(question_target):
    # Look for options to the answer in the stuff...
    world_text = []
    for this_stuff in all_the_stuff_in_the_world:
        if question_target in this_stuff:
            world_text = deepcopy(this_stuff)
            break
    if len(world_text) == 0:
        err_str = f"Cannot find stuff to make the options for target: {question_target}"
        raise ValueError(err_str)
    
    # Remove correct answer from list
    options_text = list()
    for text in world_text:
        if text != question_target:
            options_text.append(text)
    
    # Add unknowns
    options_text += ["unknown", 
                     "it is uncertain", 
                     "it is impossible to know", 
                     "not enough information", 
                     "it's impossible to know", 
                     "don't know"]
    
    
    return options_text

# --------------------- bAbI world actors and places ---------------------------
container_objects =[
    "box",
    "crate",
    "basket",
    "suitcase",
    "treasure chest",
    "box of chocolates",
    "chocolate"
]
world_actors =[
    "John",
    "Mary",
    "Sandra",
    "Daniel",
]
world_actors_2 =[
    "Jason",
    "Antoine",
    "Sumit",
    "Yann",
]
objects_moveable = [
    "nothing",
    "apple",
    "banana",
    "orange",
    "pineapple",
    "pear",
    "melon",
    "table",
    "milk",
    "football",
    "pajamas",
]
locations =[
    "office",
    "bathroom",
    "hallway",
    "garden",
    "kitchen",
    "bedroom",
]
motivations = [
    "hungry",
    "thirsty",
    "bored",
    "tired",
]
deduction_stuff = [
    "mouse",
    "sheep",
    "wolf",
    "cat",
]
deduction_plurals = {
    "mouse": "mice",
    "sheep": "sheep",
    "wolf": "wolves",
    "cat": "cats",
}
deduction_actors = [
    "Gertrude",
    "Winona",
    "Jessica",
    "Emily",
]
induction_animal = [
    'swan', 'lion', 'frog', 'rhino'
]
induction_color = ['gray', 'white', 'yellow', 'green', 'red', 'blue', 'pink']
induction_actor = ['Lily', 'Bernhard', 'Greg', 'Julius', 'Brian']
shapes = ['square', 'rectangle', 'triangle', 'sphere']
times_list = ['yesterday', 'this morning', 'this afternoon', 'this evening']
directions = [' '.join(p) for p in itertools.product(["north", "south", "east", "west"], repeat=2)]
polar = ["yes", "no"]
more_actors_task5 = [
    "Fred",
    "Jeff",
    "Bill",
    "Mary",
    "Julie",
]
more_places_task14 = [
    "cinema",
    "bedroom",
    "kitchen",
    "school",
    "office"
]
numbers = [
    "none",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
]

object_pairs = [','.join(p) for p in itertools.product([x for x in objects_moveable if x != "nothing"], repeat=2)]
object_pairs += objects_moveable # Add singles too

all_the_stuff_in_the_world = [
    container_objects, 
    world_actors,
    world_actors_2,
    objects_moveable,
    locations,
    motivations,
    deduction_stuff,
    deduction_actors,
    induction_animal,
    induction_color,
    induction_actor,
    shapes,
    times_list,
    directions,
    polar,
    more_actors_task5,
    more_places_task14,
    numbers,
    object_pairs
]
for stuff in all_the_stuff_in_the_world:
    assert len(stuff) == len(np.unique(stuff)), stuff