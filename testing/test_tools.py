from scipy import spatial
from copy import deepcopy


from testing import babi
from testing import mmlu
from testing import babisteps

from a_vert import processing as a_vert



def process_lmeh_sample_question(question_json,
                     enhance, 
                     test_type, 
                     endpoint,
                     endpoint_type, 
                     method,
                     model_name=None,
                     target_groups=["correct", "wrong", "refusal", "formulation_mistake"],
                     with_options=False,
                     option_symbol="letters",
                     distance_fn = spatial.distance.cosine, 
                     query_template=None,
                     document_template=None,
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
    out_aux = a_vert.construct_candidate_groups(correct_group_text, 
                               wrong_group_text, 
                               target_groups, 
                               enhance=enhance,
                               with_options=with_options,
                               option_symbol=option_symbol,
                               correct_group_idxs=correct_group_idxs, 
                               wrong_group_idxs=wrong_group_idxs, 
                               return_references=return_references)
    if return_references :
        group_texts_dict = out_aux[0]
        track_labels = out_aux[1]
        track_groups = out_aux[2]
    else:
        group_texts_dict = out_aux

    # Process all candidate groups
    group_distances_dict, all_distances = a_vert.get_candidate_groups_embedings_ranking(model_response,
                                            group_texts_dict,
                                           endpoint,
                                           endpoint_type, 
                                            method,
                                            model_name=model_name,
                                           query_template=query_template,
                                           document_template=document_template,
                                           distance_fn = distance_fn, 
                                           grouping_method=grouping_method, 
                                           verbose=verbose,
                                           batch_size=batch_size
                                           )

    # Return data
    if return_references:
        all_texts = list()
        for key in group_texts_dict.keys():
            all_texts += group_texts_dict[key]
        distances_data = {
            "texts" : all_texts,
            "labels" : track_labels,
            "groups" : track_groups,
            "distances" : all_distances
        }
        return group_distances_dict, distances_data
    else:
        return group_distances_dict

