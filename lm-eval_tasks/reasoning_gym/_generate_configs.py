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
    "001": "gsm_symbolic",
    "002": "polynomial_equations",
    "003": "complex_arithmetic",
    "004": "simple_integration",
    "005": "intermediate_integration",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="reasoning_gym")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    ALL_TASKS = []
    for task_id, task_name in tqdm(TASKS.items()):
        split_name = f"task_{task_id}-{task_name}"
        task_name_use = split_name
        if split_name not in ALL_TASKS:
            ALL_TASKS.append(split_name)

        # In the chat template, this is used as the system prompt
        description = ""

        yaml_dict = {
            "include": base_yaml_name,
            "tag": "reasoning_gym-all",
            "task": f"reasoning_gym-{task_name_use}",
            "task_alias": task_name_use.replace("_", " ").replace("-", " - "),
            "dataset_name": task_name,
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

        _subcategories = [f"reasoning_gym-{task}" for task in ALL_TASKS]

        file_save_path = args.save_prefix_path + set + ".yaml"

        eval_logger.info(f"Saving benchmark config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": "reasoning_gym-all",
                    "task": _subcategories,
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
