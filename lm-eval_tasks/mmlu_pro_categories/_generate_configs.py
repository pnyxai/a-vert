"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


CATEGORIES = ['biology',
 'business',
 'chemistry',
 'computer science',
 'economics',
 'engineering',
 'health',
 'history',
 'law',
 'math',
 'other',
 'philosophy',
 'physics',
 'psychology']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu_pro_categories")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    ALL_TASKS = []
    for set in ["_categories", "_categories_leaderboard"]:
        for category_name in tqdm(CATEGORIES):
            split_name = f"category_{category_name}"
            task_name_use = f"mmlu_pro{set}-{split_name}"

            if task_name_use not in ALL_TASKS:
                ALL_TASKS.append(task_name_use)

            # description = f"The following are questions on the subject of : {category_name}.\n\n"

            yaml_dict = {
                "include": base_yaml_name,
                "task": task_name_use,
                "task_alias": category_name,
                "dataset_name": split_name,
                # "description": description,
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
        
        file_save_path = args.save_prefix_path + ".yaml"

        eval_logger.info(f"Saving benchmark config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"mmlu_pro{set}-all",
                    "task": ALL_TASKS,
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
