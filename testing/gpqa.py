import re

# Function to extract data from question
def extract_data(question_json):
    question = question_json['doc']['Question']
    model_response = question_json['resps'][0][0]


    question_target = preprocess(question_json['doc']["Correct Answer"])
    choices = question_json['doc']["choices"]



    correct_group_text = list()
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx in range(len(choices)):
        if choices[idx] == question_target:
            correct_group_text.append(choices[idx])
            correct_group_idxs.append(idx)
        else:
            wrong_group_text.append(choices[idx])
            wrong_group_idxs.append(idx)
    
    assert len(correct_group_idxs) == 1
    assert len(correct_group_text) == len(correct_group_idxs)
    assert len(wrong_group_idxs) == len(wrong_group_text)
    
    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs





def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text