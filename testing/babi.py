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

def extract_data(question_json):

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
 
    # Get the question
    question = question_json['doc']['question']
    # Get the model response
    model_response = question_json['resps'][0][0]
    # Get the correct answer
    correct_group_text = [question_json['doc']['answer']]
    # Look for options to the answe in the stuff...
    options_text = []
    for this_stuff in all_the_stuff_in_the_world:
        if correct_group_text[0] in this_stuff:
            options_text = deepcopy(this_stuff)
            break
    if len(options_text) == 0:
        print(correct_group_text)
        print(question_json)
        raise ValueError("Cannot find stuff to make the options!!!")
    
    # Add unknowns
    options_text += ["unknown", 
                     "it is uncertain", 
                     "it is impossible to know", 
                     "not enough information", 
                     "it's impossible to know", 
                     "don't know"]
    
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx, text in enumerate(options_text):
        if text not in correct_group_text:
            wrong_group_text.append(text)
            wrong_group_idxs.append(idx)
        else:
            correct_group_idxs.append(idx)

    if len(correct_group_idxs) < len(correct_group_text):
        for i in range(len(correct_group_text) - len(correct_group_idxs)):
            correct_group_idxs.append(-1)
    
    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs

