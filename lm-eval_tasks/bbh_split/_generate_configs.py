"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")

TASKS = {
    "1": "boolean_expressions",
    "2": "causal_judgement",
    "3": "date_understanding",
    "4": "disambiguation_qa",
    "5": "dyck_languages",
    "6": "formal_fallacies",
    "7": "geometric_shapes",
    "8": "hyperbaton",
    "9": "logical_deduction_five_objects",
    "10": "logical_deduction_seven_objects",
    "11": "logical_deduction_three_objects",
    "12": "movie_recommendation",
    "13": "multistep_arithmetic_two",
    "14": "navigate",
    "15": "object_counting",
    "16": "penguins_in_a_table",
    "17": "reasoning_about_colored_objects",
    "18": "ruin_names",
    "19": "salient_translation_error_detection",
    "20": "snarks",
    "21": "sports_understanding",
    "22": "temporal_sequences",
    "23": "tracking_shuffled_objects_five_objects",
    "24": "tracking_shuffled_objects_seven_objects",
    "25": "tracking_shuffled_objects_three_objects"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="bbh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    
    for set in ["_split"]:
        ALL_TASKS = []
        for task_id, task_name in tqdm(TASKS.items()):
            split_name = f"task_{task_id}-{task_name}"
            task_name_use = split_name
            if int(task_id)<10:
                # To keep order correctly on display screen
                task_name_use = f"task_0{task_id}-{task_name}"
            if split_name not in ALL_TASKS:
                ALL_TASKS.append(split_name)

            # In the chat template, this is used as the system prompt
            description = ""

            yaml_dict = {
                "include": base_yaml_name,
                "tag": f"bbh{set}-all",
                "task": f"bbh{set}-{task_name_use}",
                "task_alias": task_name_use.replace("_", " ").replace("-", " - "),
                "dataset_name": split_name,
                "description": description,
            }

            file_save_path = args.save_prefix_path + f"_{task_name_use}.yaml"
            eval_logger.info(f"Saving yaml for subset {task_name_use} to {file_save_path}")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_style='"',
                )


        bbh_subcategories = [f"bbh{set}-{task}" for task in ALL_TASKS]

        
        file_save_path = args.save_prefix_path + set + ".yaml"

        eval_logger.info(f"Saving benchmark config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"bbh{set}-all",
                    "task": bbh_subcategories,
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
