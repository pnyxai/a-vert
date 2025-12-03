import re
import itertools
import numpy as np
from copy import deepcopy

import a_vert

# Setup A-VERT configuration from environment variables
AVERT_CONFIG = a_vert.setup()

# For backward compatibility, extract individual values
ENHANCE = AVERT_CONFIG.enhance

# Default instruction map
default_instruction = {
    "default": "Find the document that better represents the meaning in the query. Check for any doubts about the question or options. Focus on exact numbers, dates, or symbols.",
}
if not AVERT_CONFIG.instruction_map:
    AVERT_CONFIG.instruction_map = default_instruction

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



def doc_eval(pred, refs, question, task):
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
    correct_group_text, wrong_group_text = get_babi_options(refs, question, task)
    # Construct the wrong candidates group
    group_texts_dict = a_vert.processing.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               ["correct", "wrong"], 
                               enhance=ENHANCE,
                               )

    # Process all candidate groups
    response_group_distribution, _ = a_vert.processing.get_candidate_groups_embedings_ranking(
        pred,
        group_texts_dict,
        AVERT_CONFIG,
        task=str(task) if task is not None else "default",
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

    # Assert we are evaluating a single target. This is a limitation of this 
    # bAbI implementation
    assert len(results) == 1, "only single predictions are supported"

    
    # Get the data
    response = results[0]
    target = doc["answer"]
    question = doc["question"]
    task = doc.get("task", "default")

    # Evaluate the document with the given model response
    result_dict = doc_eval(response, target, question, task=task)

    return result_dict



# ------------------------------------------------------------------------------
# --------------------- bAbI specific code -------------------------------------
# ------------------------------------------------------------------------------

def get_babi_options(question_target, question, task):

    # Check if this is a list
    if ',' in question_target and task == 8:
        correct_group_text = [','.join(p) for p in itertools.product(question_target.split(','), repeat=2)]
        correct_group_text = [pair for pair in correct_group_text if pair.split(',')[0] != pair.split(',')[1]]
    else:
        correct_group_text = [question_target]
    
    # Look for options to the answer in the stuff...
    world_text = []
    for this_stuff in all_the_stuff_in_the_world:
        for this_correct in correct_group_text:
            if this_correct in this_stuff:
                world_text = deepcopy(this_stuff)
                break
        if len(world_text)>0:
            break
    if len(world_text) == 0:
        err_str = f"Cannot find stuff to make the options for target: {question_target}"
        raise ValueError(err_str)
    
    # Remove correct answer from list
    options_text = list()
    for text in world_text:
        if text != question_target:
            options_text.append(text)
    
    # Patching options
    if task == 4:
        # The answers to this question must be unique elements.
        options_text += [f"the {items[0]} and the {items[1]}" for items in itertools.combinations(options_text, r=2)]
        # This is a missing choice
        options_text += ["there is nothing"]

    # Add unknowns
    unknowns = ["unknown", 
                     "it is uncertain", 
                     "it is impossible to know", 
                     "not enough information", 
                     "it's impossible to know", 
                     "don't know"]
    options_text += unknowns


    # Create enhanced lists for babi
    wrong_group_text = list()
    for idx, text in enumerate(options_text):
        if text not in correct_group_text:
            wrong_group_text.append(text)

    # Patch some special cases
    question = question.lower()
    if task == 3:
        assert "where was the " in question
        assert " before the " in question

        thing, place = question.split("where was the ")[-1].split(" before the ")
        place = place[:-1]

        # Patch correct
        correct_group_text.append(f"the {thing} was in {question_target} before {place}")
        # Patch wrongs
        new_wrongs = list()
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                new_wrongs.append(f"the {thing} was in {wrong} before {place}")
        wrong_group_text += new_wrongs
    
    elif task == 4:

        if " of?" in question:
            thing, direction = question.split("what is the ")[-1].split(" of?")[0].split(" ")
            # Patch correct
            correct_group_text.append(f"the {thing} is {direction} of the {question_target}")
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{thing} is {direction} of {wrong}")
            wrong_group_text += new_wrongs
                   
        elif " of the " in question:
            direction, thing = question.split("what is ")[-1].split(" of the ")
            thing = thing[:-1]

            # Patch correct
            correct_group_text.append(f"the {question_target} is {direction} of the {thing}")
            
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    if " and the " in wrong:
                        new_wrongs.append(f"{wrong} are {direction} of the {thing}")
                    else:
                        new_wrongs.append(f"the {wrong} is {direction} of the {thing}")
            wrong_group_text += new_wrongs
        else:
            raise ValueError("question not supported in task 4!")

    
        
    elif task == 5:
        if "what did " in question:
            sub1, sub2 = question.split("what did ")[-1].split(" give to ")
            sub2 = sub2[:-1]
            # Correct
            correct_group_text.append(f"{sub1} gave the {question_target} to {sub2}")
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{sub1} gave the {wrong} to {sub2}")
            wrong_group_text += new_wrongs
        elif "who gave the " in question:
            if " to " in question:
                obj, subj = question.split("who gave the ")[-1].split(" to ")
                subj = subj[:-1]
                # Correct
                correct_group_text.append(f"{question_target} gave the {obj} to {subj}")
                # Patch wrongs
                new_wrongs = list()
                for idx, wrong in enumerate(wrong_group_text):
                    if wrong not in unknowns:
                        new_wrongs.append(f"{wrong} gave the {obj} to {subj}")
                wrong_group_text += new_wrongs
            else:
                obj = question.split("who gave the ")[-1][:-1]
                # Correct
                correct_group_text.append(f"{question_target} gave the {obj}")
                # Patch wrongs
                new_wrongs = list()
                for idx, wrong in enumerate(wrong_group_text):
                    if wrong not in unknowns:
                        new_wrongs.append(f"{wrong} gave the {obj}")
                wrong_group_text += new_wrongs
        elif "who did " in question:
            subj, obj = question.split("who did ")[-1].split(" give the ")
            obj = obj.split(" to?")[0]
            # Correct
            correct_group_text.append(f"{question_target} gave the {obj} to {subj}")
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{wrong} gave the {obj} to {subj}")
            wrong_group_text += new_wrongs
        elif "who received the " in question:
            obj = question.split("who received the ")[-1]
            obj = obj[:-1]
            # Correct
            correct_group_text.append(f"{question_target} received the {obj}")
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{wrong} received the {obj}")
            wrong_group_text += new_wrongs
        else:
            raise ValueError("Unsupported question in task 5")


    elif task == 15:
        assert "afraid" in question
        correct_group_text[0] = f"afraid of {correct_group_text[0]}"
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                wrong_group_text[idx] = f"afraid of {wrong}"

    elif task == 10 or task == 17:

        if "is the " in question:
            # For 17
            placement = question.split("is the ")[-1][:-1]
        else:
            # For 10
            placement = question.split("is ")[-1][:-1]

        if correct_group_text[0] == "yes":
            correct_group_text[0] = f"yes, the placement: {placement}, is correct"
        else:
            correct_group_text[0] = f"no, the placement: {placement}, is not correct"
        
        for idx, wrong in enumerate(wrong_group_text):
            if wrong == "yes":
                wrong_group_text[idx] = f"yes, the placement: {placement}, is correct"
            elif wrong == "no":
                wrong_group_text[idx] = f"no, the placement: {placement}, is not correct"

    elif task == 19:
        assert "how do you go from the " in question
        place1, place2 = question.split("how do you go from the ")[-1].split(" to the ")
        place2 = place2[:-1]

        t1, t2 = question_target.split(" ")
        # Correct
        correct_group_text.append(f"to go from the {place1} to the {place2} you first go {t1} and then {t2}")
        # Patch wrongs
        new_wrongs = list()
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                if " " in wrong:
                    w1, w2 = wrong.split(" ")
                    new_wrongs.append(f"to go from the {place1} to the {place2} you first go {w1} and then {w2}")
                else:
                    new_wrongs.append(f"to go from the {place1} to the {place2} you need to go {wrong}")
        wrong_group_text += new_wrongs
    
    
    return correct_group_text, wrong_group_text

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
    "park"
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
directions = ["north", "south", "east", "west"]
directions += [' '.join(p) for p in itertools.product(directions, repeat=2)]
polar = ["yes", "no", "maybe"]
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

object_pairs = [",".join(items) for items in itertools.combinations([x for x in objects_moveable if x != "nothing"], r=2)]
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