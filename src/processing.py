import numpy as np
from copy import deepcopy
from scipy import spatial


from src import embedding_tools as emb
from src import prompts_general as prompts


def get_candidate_groups_embedings_ranking(model_response,
                                           correct_group_text, 
                                           wrong_group_text, 
                                           refusal_group_text, 
                                           formulation_mistake_group_text,
                                           tei_endpoint,
                                           instruction=None,
                                           distance_fn = spatial.distance.cosine, 
                                           grouping_method="max", 
                                           batch_size=32,
                                           verbose=False,
                                           ):
    """This function takes a four lists:
    - `correct_group_text` : The list of candidates that represent a correct
                             answer.
    - `wrong_group_text` : The list of candidates that represent a wrong answer.
    - `refusal_group_text` : The list of candidates that represent a refusal
                             to answer from the language model.
    - `formulation_mistake_group_text` : The list of candidates that represent a
                             critique to the question by the language model and
                             states that the question cannot be answered.
    Then takes the language model response (`model_response`) and compares the 
    embedding of this response to each candidate group and produces a 
    classification of the model response into one of these four candidate groups
    by means of the embeddings and a distance metrics plus an aggregation 
    method.
    The result of this function is a vector (`response_group_distribution`) that
    adds up to one, where the highest value represent the highest affinity of 
    the model response to the given group, in the same order as the groups are 
    provided to this function.
    """

    # Create batch for the embedding endpoint
    batch = list()
    # Correct
    batch += correct_group_text
    idxs_correct = [0, len(correct_group_text)]
    # Wrong
    batch += wrong_group_text
    idxs_wrong = [idxs_correct[1], idxs_correct[1]+len(wrong_group_text)]
    # Refusal group
    batch += refusal_group_text
    idxs_refusal = [idxs_wrong[1], idxs_wrong[1]+len(refusal_group_text)]
    # Formulation Mistake
    batch += formulation_mistake_group_text
    idxs_formulation_mistake = [idxs_refusal[1], idxs_refusal[1]+len(formulation_mistake_group_text)]        
    

    # Calculate targets embeddings
    targets_embeddings = emb.get_embedding(batch, tei_endpoint, max_batch_size=batch_size)
    # Get model response embedding
    if instruction is not None:
        model_response_to_embedding = f'Instruct: {instruction}\nQuery:{model_response}'
    else:
        model_response_to_embedding = model_response
    model_response_embedding = np.squeeze(emb.get_embedding(model_response, tei_endpoint, max_batch_size=batch_size))

    # Calculate the distances
    all_distances = [1 - distance_fn(model_response_embedding, this_emb) for this_emb in targets_embeddings]

    # Split the distances into the corresponding groups
    correct_group_distances = all_distances[idxs_correct[0]:idxs_correct[1]]
    refusal_group_distances = all_distances[idxs_refusal[0]:idxs_refusal[1]]
    formulation_mistake_distances = all_distances[idxs_formulation_mistake[0]:idxs_formulation_mistake[1]]
    wrong_group_distances = all_distances[idxs_wrong[0]:idxs_wrong[1]]

    # Get per-group scores
    if grouping_method == "max":
        grouping_method_fn = np.max
    elif grouping_method == "mean":
        grouping_method_fn = np.mean
    else:
        raise ValueError("Grouping method not supported")
    
    refusal_score = grouping_method_fn(refusal_group_distances)
    formulation_mistake_score = grouping_method_fn(formulation_mistake_distances)
    correct_score = grouping_method_fn(correct_group_distances)
    wrong_score = grouping_method_fn(wrong_group_distances)

    # Normalize scores
    scores_all = [correct_score, wrong_score, refusal_score, formulation_mistake_score]
    response_group_distribution = scores_all/np.sum(scores_all)

    if verbose:
        print("Correct")
        for idx in np.argsort(correct_group_distances)[::-1]:
            print(f"{correct_group_distances[idx]}\t{correct_group_text[idx]}")
        print("Wrong")
        for idx in np.argsort(wrong_group_distances)[::-1]:
            print(f"{wrong_group_distances[idx]}\t{wrong_group_text[idx]}")
        print("Refusal")
        for idx in np.argsort(refusal_group_distances)[::-1]:
            print(f"{refusal_group_distances[idx]}\t{refusal_group_text[idx]}")
        print("Formulation Mistake")
        for idx in np.argsort(formulation_mistake_distances)[::-1]:
            print(f"{formulation_mistake_distances[idx]}\t{formulation_mistake_group_text[idx]}")


    return response_group_distribution, all_distances



def correct_candidate_group_construction(correct_group_text, 
                                         correct_group_idxs, 
                                         wrong_group_text, 
                                         wrong_group_idxs, 
                                         enhance, 
                                         with_options=False, 
                                         option_symbol="",
                                         return_references=False):
    """This function takes as input the correct and wrong answers and produces
    an enhanced version of the correct targets (if selected). Also, if the 
    questions has options, it will create specific candidates for the correct 
    answer.
    Optionally it will return the list of enhancements and the target groups for
    each entry in the returned text group.
    """

    if return_references :       
        # Set tracking lists, used only if return_references==True
        track_labels = list()
        track_groups = list()
        
        for i in range(len(correct_group_text)):
            track_labels.append(f"correct_reference")
            track_groups.append(f"correct")

    if enhance:
        # Basic prompt enhancing
        original_correct_group_text = deepcopy(correct_group_text)
        correct_group_text = prompts.enhance_group(correct_group_text, 
                                                   correct_group_idxs, 
                                                   with_options=with_options, 
                                                   option_symbol=option_symbol, 
                                                   return_references=return_references)
        # Track enhancements
        if return_references :
            labels = correct_group_text[1]
            correct_group_text = correct_group_text[0]
            for l in labels:
                track_labels.append(l)
                track_groups.append(f"correct")

        # Enhance options
        if with_options:
            assert len(original_correct_group_text) == 1, "Cannot have multiple correct candidates in a multiple choice question and use options enhancement."
            # Get enhancements specific for multiple choice questions
            correct_group_text_aux = prompts.all_options_group(original_correct_group_text[-1], 
                                                            correct_group_idxs[-1], 
                                                            wrong_group_text, 
                                                            wrong_group_idxs,
                                                            return_references=return_references)
            # Track enhancements
            if return_references :
                    labels = correct_group_text_aux[1]
                    correct_group_text_aux = correct_group_text_aux[0]
                    for l in labels:
                        track_labels.append(l)
                        track_groups.append(f"correct")
            correct_group_text += correct_group_text_aux

    if return_references :
        return correct_group_text, track_labels, track_groups
    else:
        return correct_group_text

def wrong_candidate_group_construction(correct_group_text, 
                                         correct_group_idxs, 
                                         wrong_group_text, 
                                         wrong_group_idxs, 
                                         enhance, 
                                         with_options=False, 
                                         option_symbol="",
                                         return_references=False):
    """This function takes as input the correct and wrong answers and produces
    an enhanced version of the wrong targets (if selected). Also, if the 
    questions has options, it will create specific candidates for the correct 
    answer.
    Optionally it will return the list of enhancements and the target groups for
    each entry in the returned text group.
    """


    if return_references :
        # Set tracking lists, used only if return_references==True
        track_labels = list()
        track_groups = list()

        for i in range(len(wrong_group_text)):
            track_labels.append(f"wrong_reference")
            track_groups.append(f"wrong")

    if enhance:
        # Basic prompt enhancing
        original_wrong_group_text = deepcopy(wrong_group_text)
        wrong_group_text = prompts.enhance_group(wrong_group_text, 
                                                 wrong_group_idxs, 
                                                 with_options=with_options, 
                                                 option_symbol=option_symbol, 
                                                 return_references=return_references)
        
        # Track enhancements
        if return_references :
            labels = wrong_group_text[1]
            wrong_group_text = wrong_group_text[0]
            for l in labels:
                track_labels.append(l)
                track_groups.append(f"wrong")
        
        if with_options:
            # All wrong permutations
            for this_idx in range(len(wrong_group_idxs)):
                fake_wrongs_texts = original_wrong_group_text[:this_idx]
                fake_wrongs_idxs = wrong_group_idxs[:this_idx]
                if this_idx+1 < len(wrong_group_idxs):
                    fake_wrongs_texts += original_wrong_group_text[this_idx+1:]
                    fake_wrongs_idxs += wrong_group_idxs[this_idx+1:]
                fake_wrongs_texts += correct_group_text
                fake_wrongs_idxs += correct_group_idxs
                wrong_group_text_aux = prompts.all_options_group(original_wrong_group_text[this_idx], 
                                                      wrong_group_idxs[this_idx],
                                                      fake_wrongs_texts,
                                                      fake_wrongs_idxs,
                                                      return_references=return_references)
                # Track enhancements
                if return_references :
                    labels = wrong_group_text_aux[1]
                    wrong_group_text_aux = wrong_group_text_aux[0]
                    for l in labels:
                        track_labels.append(l)
                        track_groups.append(f"wrong")
                
                wrong_group_text += wrong_group_text_aux

    if return_references :
        return wrong_group_text, track_labels, track_groups
    else:
        return wrong_group_text
    


def refusal_candidate_group_construction(return_references=False):
    """This function produces a list of refusal candidates. Refusals represent
    the cases where the language model chooses not to answer the question.
    Optionally it will return the list of enhancements and the target groups for
    each entry in the returned text group.
    """

    # Get base candidates
    refusal_group_text = prompts.refusal_group_text
    if return_references :
        # Set tracking lists, used only if return_references==True
        track_labels = list()
        track_groups = list()
        for i in range(len(refusal_group_text)):
            track_labels.append(f"refusal_{i+1}")
            track_groups.append(f"refusal")

    if return_references :
        return refusal_group_text, track_labels, track_groups
    else:
        return refusal_group_text
    

def question_mistake_candidate_group_construction(with_options, return_references=False):
    """This function produces a list of question mistake candidates. these
    represent the cases where the language model says that the presented 
    question is wrong and therefore it cannot be answered.
    Optionally it will return the list of enhancements and the target groups for
    each entry in the returned text group.
    """

    # Get base candidates
    formulation_mistake_group_text = deepcopy(prompts.formulation_mistake_base_group_text)
    # Track enhancements
    if return_references :
        # Set tracking lists, used only if return_references==True
        track_labels = list()
        track_groups = list()
        for i in range(len(formulation_mistake_group_text)):
            track_labels.append(f"formulation_mistake_{i+1}")
            track_groups.append(f"formulation_mistake")

    if with_options:
        # Add option-specific cases to the refusal candidates
        current_len = len(formulation_mistake_group_text)
        formulation_mistake_group_text += prompts.formulation_mistake_choices_group_text
        # Track enhancements
        if return_references :
            enhanced_num = len(formulation_mistake_group_text)-current_len
            for i in range(enhanced_num):
                track_labels.append(f"formulation_mistake_options_{i+1}")
                track_groups.append(f"formulation_mistake")

    if return_references :
        return formulation_mistake_group_text, track_labels, track_groups
    else:
        return formulation_mistake_group_text
    



    
    