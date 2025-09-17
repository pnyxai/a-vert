
import numpy as np

# Function to extract data from question
def extract_data(question_json):
    question = question_json['doc']['question']
    model_response = question_json['resps'][0][0]

    question_target = question_json['doc']['answer']



    # Get target number
    target_num = int(question_target.split("#### ")[-1])
    # Set other numbers
    other_options = [
        np.floor(target_num*0.1),
        np.floor(target_num*0.5),
        np.ceil(target_num*1.25),
        np.ceil(target_num*1.8),
        target_num+1,
        target_num-1
    ]
    other_options = np.unique(other_options)
    other_options = [int(a) for a in other_options if a != target_num]

    wrong_group_text = [f"{a}" for a in other_options]
    correct_group_text = [f"{target_num}"]


    
    wrong_group_idxs = list()
    for idx in range(len(wrong_group_text)):
        wrong_group_idxs.append(idx)
    correct_group_idxs = [len(wrong_group_idxs)]
    
    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs