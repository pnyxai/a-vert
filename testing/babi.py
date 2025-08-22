import numpy as np
from copy import deepcopy
import itertools

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
directions = ["north", "south", "east", "west"]
directions += [' '.join(p) for p in itertools.product(["north", "south", "east", "west"], repeat=2)]
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

def extract_data(question_json):

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
 
    # Get the model response
    model_response = question_json['resps'][0][0]
    # Get the correct answer
    this_answer = question_json['doc']['answer']
    # Check if this is a list
    if ',' in this_answer and question_json['doc']['task'] == 8:
        correct_group_text = [','.join(p) for p in itertools.product(this_answer.split(','), repeat=2)]
        correct_group_text = [pair for pair in correct_group_text if pair.split(',')[0] != pair.split(',')[1]]
    else:
        correct_group_text = [this_answer]
    # Look for options to the answer in the stuff...
    options_text = []
    for this_stuff in all_the_stuff_in_the_world:
        for this_correct in correct_group_text:
            if this_correct in this_stuff:
                options_text = deepcopy(this_stuff)
                break
        if len(options_text)>0:
            break
    if len(options_text) == 0:
        print(correct_group_text)
        print(question_json)
        raise ValueError("Cannot find stuff to make the options!!!")
    
    # Patching options
    if question_json['doc']['task'] == 4:
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
    

    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx, text in enumerate(options_text):
        if text not in correct_group_text:
            wrong_group_text.append(text)
            wrong_group_idxs.append(idx)
        else:
            correct_group_idxs.append(idx)


    # Patch some special cases
    target = correct_group_text[0]
    question = question_json['doc']['question'].lower()
    
    if question_json['doc']['task'] == 3:
        assert "where was the " in question
        assert " before the " in question

        thing, place = question.split("where was the ")[-1].split(" before the ")
        place = place[:-1]

        # Patch correct
        correct_group_text.append(f"the {thing} was in {target} before {place}")
        correct_group_idxs.append(correct_group_idxs[-1]+1)
        # Patch wrongs
        new_wrongs = list()
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                new_wrongs.append(f"the {thing} was in {wrong} before {place}")
                wrong_group_idxs.append(wrong_group_idxs[-1]+1)
        wrong_group_text += new_wrongs
    
    elif question_json['doc']['task'] == 4:

        if " of?" in question:
            thing, direction = question.split("what is the ")[-1].split(" of?")[0].split(" ")
            # Patch correct
            correct_group_text.append(f"the {thing} is {direction} of the {target}")
            correct_group_idxs.append(correct_group_idxs[-1]+1)
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{thing} is {direction} of {wrong}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
            wrong_group_text += new_wrongs
                   
        elif " of the " in question:
            direction, thing = question.split("what is ")[-1].split(" of the ")
            thing = thing[:-1]

            # Patch correct
            correct_group_text.append(f"the {target} is {direction} of the {thing}")
            correct_group_idxs.append(correct_group_idxs[-1]+1)
            
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    if " and the " in wrong:
                        new_wrongs.append(f"{wrong} are {direction} of the {thing}")
                    else:
                        new_wrongs.append(f"the {wrong} is {direction} of the {thing}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
            wrong_group_text += new_wrongs
        else:
            raise ValueError("question not supported in task 4!")

    
        
    elif question_json['doc']['task'] == 5:
        if "what did " in question:
            sub1, sub2 = question.split("what did ")[-1].split(" give to ")
            sub2 = sub2[:-1]
            # Correct
            correct_group_text.append(f"{sub1} gave the {target} to {sub2}")
            correct_group_idxs.append(correct_group_idxs[-1]+1)
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{sub1} gave the {wrong} to {sub2}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
            wrong_group_text += new_wrongs
        elif "who gave the " in question:
            if " to " in question:
                obj, subj = question.split("who gave the ")[-1].split(" to ")
                subj = subj[:-1]
                # Correct
                correct_group_text.append(f"{target} gave the {obj} to {subj}")
                correct_group_idxs.append(correct_group_idxs[-1]+1)
                # Patch wrongs
                new_wrongs = list()
                for idx, wrong in enumerate(wrong_group_text):
                    if wrong not in unknowns:
                        new_wrongs.append(f"{wrong} gave the {obj} to {subj}")
                        wrong_group_idxs.append(wrong_group_idxs[-1]+1)
                wrong_group_text += new_wrongs
            else:
                obj = question.split("who gave the ")[-1][:-1]
                # Correct
                correct_group_text.append(f"{target} gave the {obj}")
                correct_group_idxs.append(correct_group_idxs[-1]+1)
                # Patch wrongs
                new_wrongs = list()
                for idx, wrong in enumerate(wrong_group_text):
                    if wrong not in unknowns:
                        new_wrongs.append(f"{wrong} gave the {obj}")
                        wrong_group_idxs.append(wrong_group_idxs[-1]+1)
                wrong_group_text += new_wrongs
        elif "who did " in question:
            subj, obj = question.split("who did ")[-1].split(" give the ")
            obj = obj.split(" to?")[0]
            # Correct
            correct_group_text.append(f"{target} gave the {obj} to {subj}")
            correct_group_idxs.append(correct_group_idxs[-1]+1)
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{wrong} gave the {obj} to {subj}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
            wrong_group_text += new_wrongs
        elif "who received the " in question:
            obj = question.split("who received the ")[-1]
            obj = obj[:-1]
            # Correct
            correct_group_text.append(f"{target} received the {obj}")
            correct_group_idxs.append(correct_group_idxs[-1]+1)
            # Patch wrongs
            new_wrongs = list()
            for idx, wrong in enumerate(wrong_group_text):
                if wrong not in unknowns:
                    new_wrongs.append(f"{wrong} received the {obj}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
            wrong_group_text += new_wrongs
        else:
            raise ValueError("Unsupported question in task 5")


    elif question_json['doc']['task'] == 15:
        assert "afraid" in question
        correct_group_text[0] = f"afraid of {correct_group_text[0]}"
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                wrong_group_text[idx] = f"afraid of {wrong}"

    elif question_json['doc']['task'] == 10 or question_json['doc']['task'] == 17:

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

    elif question_json['doc']['task'] == 19:
        assert "how do you go from the " in question
        place1, place2 = question.split("how do you go from the ")[-1].split(" to the ")
        place2 = place2[:-1]

        t1, t2 = target.split(" ")
        # Correct
        correct_group_text.append(f"to go from the {place1} to the {place2} you first go {t1} and then {t2}")
        correct_group_idxs.append(correct_group_idxs[-1]+1)
        # Patch wrongs
        new_wrongs = list()
        for idx, wrong in enumerate(wrong_group_text):
            if wrong not in unknowns:
                if " " in wrong:
                    w1, w2 = wrong.split(" ")
                    new_wrongs.append(f"to go from the {place1} to the {place2} you first go {w1} and then {w2}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
                else:
                    new_wrongs.append(f"to go from the {place1} to the {place2} you need to go {wrong}")
                    wrong_group_idxs.append(wrong_group_idxs[-1]+1)
        wrong_group_text += new_wrongs



    if len(correct_group_idxs) < len(correct_group_text):
        for i in range(len(correct_group_text) - len(correct_group_idxs)):
            correct_group_idxs.append(-1)
    
    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs

