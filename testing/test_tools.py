from scipy import spatial
from copy import deepcopy


from testing import babi
from testing import mmlu
from testing import babisteps

from src import processing as a_vert



def process_lmeh_sample_question(question_json,
                     enhance, 
                     test_type, 
                     tei_endpoint,
                     with_options=False,
                     option_symbol="letters",
                     distance_fn = spatial.distance.cosine, 
                     instruction=None, 
                     grouping_method="max", 
                     batch_size=32,
                     verbose=False,
                     return_references=False):
    """Testing function that takes as input a json containing the lm-eval sample 
    data and classifies the answer using the a-vert method.
    """

    # Assign the data extraction depending on the dataset used
    if test_type == "mmlu":
        extract_data = mmlu.extract_data
    elif test_type == "babisteps":
        extract_data = babisteps.extract_data
    elif test_type == "babi":
        extract_data = babi.extract_data
    else:
        raise ValueError("Dataset type not supported.")
    
    # Extract data
    (question, 
     model_response, 
     correct_group_text, 
     wrong_group_text, 
     correct_group_idxs, 
     wrong_group_idxs) = extract_data(question_json)
    
    assert len(correct_group_text)==len(correct_group_idxs), f"{len(correct_group_text)} -- {len(correct_group_idxs)}"
    assert len(wrong_group_text)==len(wrong_group_idxs), f"{len(wrong_group_text)} -- {len(wrong_group_idxs)}"

    # Construct the candidates for each group: Correct, Wrong, Refusal and Formulation mistake
    track_labels = list()
    track_groups = list()
    original_correct_group_text = deepcopy(correct_group_text)
    # original_wrong_group_text = deepcopy(wrong_group_text)
    #--------------------- CORRECT GROUP ---------------------------------------
    correct_candidates = a_vert.correct_candidate_group_construction(correct_group_text, 
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


    #--------------------- WRONG GROUP -----------------------------------------

    wrong_candidates = a_vert.wrong_candidate_group_construction(original_correct_group_text, 
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


    #--------------------- REFUSAL GROUP ---------------------------------------

    refusal_candidates = a_vert.refusal_candidate_group_construction(return_references=return_references)
    if return_references :
        refusal_group_text, track_labels_aux, track_groups_aux = refusal_candidates
        track_labels += track_labels_aux
        track_groups += track_groups_aux
    else:
        refusal_group_text = refusal_candidates

   

    #--------------------- FORMULATION MISTAKE GROUP ---------------------------

    question_mistake_candidates = a_vert.question_mistake_candidate_group_construction(with_options, return_references=return_references)
    if return_references :
        formulation_mistake_group_text, track_labels_aux, track_groups_aux = question_mistake_candidates
        track_labels += track_labels_aux
        track_groups += track_groups_aux
    else:
        formulation_mistake_group_text = question_mistake_candidates


    # Process all candidate groups
    response_group_distribution, all_distances = a_vert.get_candidate_groups_embedings_ranking(model_response,
                                           correct_group_text, 
                                           wrong_group_text, 
                                           refusal_group_text, 
                                           formulation_mistake_group_text,
                                           tei_endpoint,
                                           instruction=instruction,
                                           distance_fn = distance_fn, 
                                           grouping_method=grouping_method, 
                                           verbose=verbose,
                                           batch_size=batch_size
                                           )

    # Return data
    if return_references:
        distances_data = {
            "texts" : correct_group_text+wrong_group_text+refusal_group_text+formulation_mistake_group_text,
            "labels" : track_labels,
            "groups" : track_groups,
            "distances" : all_distances
        }
        return response_group_distribution, distances_data
    else:
        return response_group_distribution

