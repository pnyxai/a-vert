import re
import itertools
import numpy as np
from copy import deepcopy
import os
from scipy import spatial

# TODO: Replace with package!
import sys
sys.path.append('/')
from a_vert import processing as a_vert

# This environment variable should contain an endpoint of https://github.com/huggingface/text-embeddings-inference
# (or compatible one) loaded with the desired embedding model.
# For example `export TEI_ENDPOINT=http://127.0.0.1:8080/embed`
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", None)
if TEI_ENDPOINT is None:
    raise ValueError("TEI_ENDPOINT environment variable is not set. This is required for A-VERT to function.")

# Instruction to be provided to the embedding of the model response. The "query" field will be replaced by the LM response.
INSTRUCTION = instruction='Instruct: Extract the conclusion or final answer. Focus on exact numbers, dates, or symbols.\nQuery:{query}'

# Grouping to be applied to candidate groups
GROUPING="max"

# Distance function used to compare embeddings
DISTANCE_FN = spatial.distance.cosine

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
    response_group_distribution, asd = a_vert.get_candidate_groups_embedings_ranking([pred],
                                           group_texts_dict,
                                           TEI_ENDPOINT,
                                           instruction=INSTRUCTION,
                                           distance_fn = DISTANCE_FN, 
                                           grouping_method=GROUPING, 
                                           verbose=False
                                           )

    # Check if this is a match
    a_vert_match = True
    if response_group_distribution["correct"] < response_group_distribution["wrong"]:
        a_vert_match = False

    # --------------------------------------------------------------------------

    # Compile and return
    results = {
        "exact_match": exact_match,
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