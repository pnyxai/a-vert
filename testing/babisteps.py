def extract_data_contextualized(question_json):

    question = question_json['doc']['question']
    model_response = question_json['resps'][0][0]

    correct_group_text = question_json['doc']['contextualized_answer']
    wrong_group_text = list()
    correct_group_idxs = list()
    wrong_group_idxs = list()
    for idx, option in enumerate(question_json['doc']['contextualized_options'] ):
        add = True
        for answ in correct_group_text:
            if answ == option:
                add = False
        if add:
            wrong_group_text.append(option)  
            wrong_group_idxs.append(idx)
        else:
            correct_group_idxs.append(idx) 
    if len(correct_group_idxs) < len(correct_group_text):
        for i in range(len(correct_group_text) - len(correct_group_idxs)):
            correct_group_idxs.append(-1)    

    return question, model_response, correct_group_text, wrong_group_text, correct_group_idxs, wrong_group_idxs

def extract_data(question_json):
    question = question_json['doc']['question']
    model_response = question_json['resps'][0][0]

    correct_group_text = question_json['doc']['answer']
    options_text = question_json['doc']['options']

    if isinstance(options_text[0], list):
        def convert2str(lista):
            if len(lista) == 1:
                return lista[0]
            elif len(lista) == 2:
                return f"{lista[0]} and {lista[1]}"
            else:
                return ", ".join(lista[:-1]) + f" and {lista[-1]}"
        # Conver enumeration lists to plain text
        options_text_plain = list()
        for lista in options_text:
            options_text_plain.append(convert2str(lista))
        options_text = options_text_plain

        if question_json['doc']["leaf_label"] == "unknown" or question_json['doc']["leaf_label"] == "none":
            # It is ok just like this
            pass
        else:
            correct_group_text = [convert2str(correct_group_text)]
        


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
