import random
import re
from functools import partial

from lm_eval.api.registry import get_metric, register_filter
from lm_eval.filters.extraction import Filter


###########################################################
## Filters
###########################################################
@register_filter("replace_regex")  # Register with the same name as before
class ReplaceFilter(Filter):
    """A filter that replaces text using regex substitution, refactored."""

    def __init__(
        self,
        regex_pattern: str,
        replacement_string: str = "",  # Default is an empty string to remove matches
        count: int = 0,  # Number of replacements to perform (0 means all)
    ) -> None:
        """
        Compiles `regex_pattern` and replaces matches with `replacement_string`.
        `count` specifies the maximum number of replacements.
        Defaults to replacing all occurrences with an empty string
        (i.e., removing them).
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.replacement_string = replacement_string
        self.count = count

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # Here, we assume resps is a list of lists, where each inner list is
        # a set of model responses for a particular input/target pair.
        # We process each of these inner lists independently.

        def filter_set(inst: list[str]) -> list[str]:
            """
            Applies the regex replacement to each response string in a
            single instance's list.
            Args:
                inst: A list of strings (responses for one input).
            Returns:
                A list of strings with replacements applied.
            """
            filtered_instance = []
            for resp in inst:  # resp is an individual string from the inner list
                # Use re.sub for replacement
                # re.sub returns a string after replacement
                processed_resp = self.regex.sub(
                    self.replacement_string, resp, count=self.count
                )
                # Append the processed string (not a list of characters)
                filtered_instance.append(processed_resp)
            # filter_set returns a list of strings for this instance
            return filtered_instance

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # filtered_resps will be a list[list[str]]
        return filtered_resps


###########################################################
# FORMAT
###########################################################


# ### Base ###
def base_format(example: dict, include_options: bool) -> str:
    prompt = ""
    prompt += "\n\n## World enumeration ##\n"
    world = example["world_enumerate"]
    prompt += world
    prompt += "\n\n## Story ##\n"
    story = example["story"]
    prompt += story
    prompt += "\nQuestion: "
    question = example["question"]
    prompt += question + "\n"
    if include_options:
        prompt += "Options:\n"
        options = example["options"]
        for i, opt in enumerate(options):
            prompt += "{}\n".format(opt)
    prompt += "Answer: "
    return prompt


def format_example(example, include_options: bool, including_answer: bool):
    prompt = base_format(example, include_options)
    if including_answer:
        answer = random.choice(example["answer"])
        prompt += answer
    return prompt


doc_to_text = partial(format_example, include_options=False, including_answer=False)
fewshot_to_text = partial(format_example, include_options=False, including_answer=True)


# ### Listing ###
def listing_format_example(example, include_options: bool, including_answer: bool):
    prompt = base_format(example, include_options)
    if including_answer:
        if example["leaf_label"] == "none" or example["leaf_label"] == "unknown":
            answer = random.choice(example["answer"])
        else:
            answer = ", ".join(example["answer"])
        prompt += answer
    return prompt


listing_doc_to_text = partial(
    listing_format_example, include_options=False, including_answer=False
)
listing_fewshot_to_text = partial(
    listing_format_example, include_options=False, including_answer=True
)


def process_results_listing(doc, results):
    result_dict = {}
    if doc["leaf_label"] == "none" or doc["leaf_label"] == "unknown":
        # In this cases, due to the answer was sampled randomly, we need to
        # check if the answer is in any of the results insted
        # of check set equality like in the else case
        metric = "exact_match"
        gold = doc["answer"]
        result = [results[0] for _ in range(len(gold))]
        scores = get_metric(metric)(
            references=gold,
            predictions=result,
        )[metric]
        exact_match = 1.0 if scores > 0.0 else 0.0
    else:
        # convert doc['answer'] into a set
        answer_set = set(doc["answer"])
        # sub spaces with empty string
        results[0] = results[0].replace(" ", "")
        # split results into a list by spliting by ", "
        results_list = results[0].split(",")
        # convert results_list into a set
        results_set = set(results_list)
        # if both sets are equal, then 1, else 0
        exact_match = 1 if answer_set == results_set else 0
    result_dict["exact_match"] = exact_match
    return result_dict
