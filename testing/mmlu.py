
# Function to extract data from question
def extract_data(question_json):
    question = question_json['doc']['question']
    model_response = question_json['resps'][0][0]

    target_idx = question_json['doc']['answer']

    correct_group_text = list()
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx in range(len(question_json['doc']['choices'])):
        if idx == target_idx:
            correct_group_text.append(question_json['doc']['choices'][idx])
            correct_group_idxs.append(idx)
        else:
            wrong_group_text.append(question_json['doc']['choices'][idx])
            wrong_group_idxs.append(idx)
    
    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs