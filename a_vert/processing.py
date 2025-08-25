import numpy as np
from copy import deepcopy
from scipy import spatial


from a_vert import embedding_tools as emb
from a_vert import prompts_general as prompts


def get_candidate_groups_embedings_ranking(model_response,
                                           candidate_groups_dict,
                                           endpoint,
                                           endpoint_type, 
                                           method,
                                           model_name=None,
                                           query_template=None,
                                           document_template=None,
                                           distance_fn = spatial.distance.cosine, 
                                           grouping_method="max", 
                                           batch_size=32,
                                           max_len=-1,
                                           verbose=False,
                                           ):
    """This function takes a dictionary of candidate groups. Each element of the
    dictionary is list of text entries to be evaluated.
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
    indexes_dict = dict()
    last_group = None
    for group_name in candidate_groups_dict.keys():
        these_texts = candidate_groups_dict[group_name]
        batch += these_texts
        if last_group is None:
            indexes_dict[group_name] = [0, len(these_texts)]
        else:
            indexes_dict[group_name] = [indexes_dict[last_group][1], indexes_dict[last_group][1]+len(these_texts)]
        last_group = group_name
    
    # Calculate semantic distances
    if method == "embedding":
        all_distances = emb.calculate_embedding_distances(model_response,
                                    batch,
                                    endpoint,
                                    endpoint_type, 
                                    model_name=model_name,
                                    query_template=query_template,
                                    document_template=document_template,
                                    distance_fn = distance_fn, 
                                    batch_size=batch_size,
                                    max_len=max_len,
                                    )
    elif method == "rerank":
        all_distances = emb.calculate_reranking_distances(model_response,
                                  batch,
                                  endpoint,
                                  endpoint_type,
                                  model_name=model_name,
                                  query_template=query_template,
                                  document_template=document_template,
                                  max_len=max_len,
                                  )
    else:
        raise ValueError("Embedding distance calculation method not supported.")
    

    # Select grouping method
    if grouping_method == "max":
        grouping_method_fn = np.max
    elif grouping_method == "mean":
        grouping_method_fn = np.mean
    elif "mean_top_k_" in grouping_method:
        top_k = int(grouping_method.split("mean_top_k_")[-1])
        grouping_method_fn = lambda x : np.mean(x[:top_k])
    else:
        raise ValueError("Grouping method not supported")

    # Split the distances into the corresponding groups
    group_distances_dict = dict()
    norm_sum = 0
    for group_name in candidate_groups_dict.keys():
        this_group_distances = all_distances[indexes_dict[group_name][0]:indexes_dict[group_name][1]]
        group_distances_dict[group_name] = grouping_method_fn(this_group_distances)
        # Track total score for normalization
        norm_sum+=group_distances_dict[group_name]

        if verbose:
            print(f"Group: {group_name}")
            for idx in np.argsort(this_group_distances)[::-1]:
                print(f"{this_group_distances[idx]}\t{candidate_groups_dict[group_name][idx]}")

    # Normalize scores
    for group_name in candidate_groups_dict.keys():
        group_distances_dict[group_name] /= norm_sum


    return group_distances_dict, all_distances





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
    


def construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               target_group_names_list, 
                               enhance=True,
                               with_options=False,
                               option_symbol=None,
                               correct_group_idxs=None, 
                               wrong_group_idxs=None, 
                               return_references=False):

    assert len(target_group_names_list) == len(np.unique(target_group_names_list)), "Group names contain duplicated elements."
    
    # Complete data, for compatibility with functions
    if not with_options and (correct_group_idxs is None or wrong_group_idxs is None):
        correct_group_idxs = [i for i in range(len(correct_group_text))]
        wrong_group_idxs = [i for i in range(len(wrong_group_text))]
    if  with_options and (correct_group_idxs is None or wrong_group_idxs is None):
        raise ValueError("If the target is with options enhancements, the targets indexes must be provided.")


    # Construct the candidates for each group: Correct, Wrong, Refusal and Formulation mistake
    track_labels = list()
    track_groups = list()
    original_correct_group_text = deepcopy(correct_group_text)
    # original_wrong_group_text = deepcopy(wrong_group_text)
    output_dict = dict()
    for group_name in target_group_names_list:

        #--------------------- CORRECT GROUP ---------------------------------------
        if group_name == "correct":
            
            correct_candidates = correct_candidate_group_construction(correct_group_text, 
                                                correct_group_idxs, 
                                                wrong_group_text, 
                                                wrong_group_idxs, 
                                                enhance, 
                                                with_options=with_options, 
                                                option_symbol=option_symbol,
                                                return_references=return_references)
            if return_references :
                correct_group_text, track_labels_aux, track_groups_aux = correct_candidates
                track_labels += track_labels_aux
                track_groups += track_groups_aux
            else:
                correct_group_text = correct_candidates

            output_dict[group_name] = correct_group_text
        #--------------------- WRONG GROUP -----------------------------------------
        elif group_name == "wrong":
            wrong_candidates = wrong_candidate_group_construction(original_correct_group_text, 
                                                correct_group_idxs, 
                                                wrong_group_text, 
                                                wrong_group_idxs, 
                                                enhance, 
                                                with_options=with_options, 
                                                option_symbol=option_symbol,
                                                return_references=return_references)
            if return_references :
                wrong_group_text, track_labels_aux, track_groups_aux = wrong_candidates
                track_labels += track_labels_aux
                track_groups += track_groups_aux
            else:
                wrong_group_text = wrong_candidates

            output_dict[group_name] = wrong_group_text
        #--------------------- REFUSAL GROUP ---------------------------------------
        elif group_name == "refusal":

            refusal_candidates = refusal_candidate_group_construction(return_references=return_references)
            if return_references :
                refusal_group_text, track_labels_aux, track_groups_aux = refusal_candidates
                track_labels += track_labels_aux
                track_groups += track_groups_aux
            else:
                refusal_group_text = refusal_candidates

            output_dict[group_name] = refusal_group_text
        #--------------------- FORMULATION MISTAKE GROUP ---------------------------
        elif group_name == "formulation_mistake":

            question_mistake_candidates = question_mistake_candidate_group_construction(with_options, return_references=return_references)
            if return_references :
                formulation_mistake_group_text, track_labels_aux, track_groups_aux = question_mistake_candidates
                track_labels += track_labels_aux
                track_groups += track_groups_aux
            else:
                formulation_mistake_group_text = question_mistake_candidates

            output_dict[group_name] = formulation_mistake_group_text
        else:
            raise ValueError(f"Group name \"{group_name}\" is not defined.")

    if return_references :
        return output_dict, track_labels, track_groups
    else:
        return output_dict
    
    