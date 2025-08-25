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
    "1": "single_supporting_fact",
    "2": "two_supporting_facts",
    "3": "three_supporting_facts",
    "4": "two_argument_relations",
    "5": "three_argument_relations",
    "6": "yes_no_questions",
    "7": "counting",
    "8": "lists_sets",
    "9": "simple_negation",
    "10": "indefinite_knowledge",
    "11": "basic_coreference",
    "12": "conjunction",
    "13": "compound_coreference",
    "14": "time_reasoning",
    "15": "basic_deduction",
    "16": "basic_induction",
    "17": "positional_reasoning",
    "18": "size_reasoning",
    "19": "path_finding",
    "20": "agents_motivations"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="babi")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    
    for set in ["_tasks", "_tasks_completions"]:
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
                "tag": f"babi{set}-all",
                "task": f"babi{set}-{task_name_use}",
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


        babi_subcategories = [f"babi{set}-{task}" for task in ALL_TASKS]

        
        file_save_path = args.save_prefix_path + set + ".yaml"

        eval_logger.info(f"Saving benchmark config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"babi{set}-all",
                    "task": babi_subcategories,
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
