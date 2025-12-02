import re
import numpy as np

# try to import reasoning_gym code, if fail raise exception
try:
    from reasoning_gym.code.codeio import CodeIODataset, CodeIOConfig
except ImportError as e:
    raise ImportError(
        "reasoning_gym package is required for this task. Please install it via `pip install reasoning-gym`."
    ) from e
import a_vert

# Setup A-VERT configuration from environment variables
AVERT_CONFIG = a_vert.setup()

# For backward compatibility, extract individual values
ENHANCE = AVERT_CONFIG.enhance

# Default instruction map
default_instruction = {
    "default": "Find the document that contians the closest numerical result or expresion in the Query.",
}
if not AVERT_CONFIG.instruction_map:
    AVERT_CONFIG.instruction_map = default_instruction

# This is a distance threshold that we will use to avoid false positives.
# Evaluating math with semantic processes is not solved by this version of
# A-VERT so, we need to be creative
DISTANCE_THRESHOLD = 0.6

def filter_response(pred):
    """This function is used by the "exact_match" metric to try to clean the
    model generated answer.
    """

    try:
        # Filter everything after the first break line
        filtered_pred = re.findall(r"^(.*?)(?=\n|$)", pred)[0].strip()
        # Remove leading white spaces
        filtered_pred = filtered_pred.lstrip()
        # function to ignore right white spaces or line breaks
        filtered_pred = re.findall(r"^(.*?)\s*$", filtered_pred)[0].strip()
    except Exception:
        filtered_pred = "[invalid]"

    return filtered_pred


def process_results(doc, results):
    """Custom processing function used to implement "a-vert" metric."""
    if doc["task"] == "codeio":
        results = process_codeio(doc, results)
    else:
        results = process_a_vert(doc, results)

    return results


# ------------------------------------------------------------------------------
# --------------------- a-vert specific code -----------------------------------
# ------------------------------------------------------------------------------


def doc_eval(pred, options, answers, question, task):
    """This function takes a model generated response ("pred") and the"""

    # ----------------------- EXACT MATCH --------------------------------------
    # Filter response
    filtered_pred = filter_response(pred)

    # Get match
    exact_match = False
    for answ in answers:
        if filtered_pred == answ:
            exact_match = True

    # ----------------------- A-VERT -------------------------------------------
    # Get other elements from the bAbI world
    correct_group_text, wrong_group_text = get_reasoning_gym_options(
        answers, question, options, task
    )
    # Construct the wrong candidates group
    group_texts_dict = a_vert.processing.construct_candidate_groups(
        correct_group_text,
        wrong_group_text,
        ["correct", "wrong"],
        enhance=ENHANCE,
    )

    # Process all candidate groups
    response_group_distribution, all_distances = (
        a_vert.processing.get_candidate_groups_embedings_ranking(
            pred,
            group_texts_dict,
            AVERT_CONFIG,
            task=task if task else "default",
        )
    )
    # Check if this is a match
    a_vert_match = True
    not_valid = np.max(all_distances) < DISTANCE_THRESHOLD
    if (
        response_group_distribution["correct"] < response_group_distribution["wrong"]
    ) or not_valid:
        a_vert_match = False

    # --------------------------------------------------------------------------

    # Compile and return
    results = {
        "exact_match": exact_match,
        "score_match": 1.0 if (a_vert_match or exact_match) else 0.0,
        "a-vert_correct_score": response_group_distribution["correct"],
        "a-vert_wrong_score": response_group_distribution["wrong"],
        "a-vert_match": a_vert_match,
        "a-vert_valid": not not_valid,
    }

    return results


def process_a_vert(doc, results):
    """Custom processing function used to implement "a-vert" metric."""

    # Assert we are evaluating a single target. This is a limitation of this
    # bAbI implementation
    assert len(results) == 1, "only single predictions are supported"

    # Get the data
    # print(doc)
    response = results[0]
    # answer = [doc["answer"]]
    # answer = [doc["answer"]]+doc["contextualized_answers"]
    answer = doc["contextualized_answers"]
    # options = [str(a) for a in doc["options"]]
    # options = [str(a) for a in doc["options"]]+doc["contextualized_options"]
    options = doc["contextualized_options"]
    question = doc["question"]
    task = doc.get("task", "default")

    # Evaluate the document with the given model response
    result_dict = doc_eval(response, options, answer, question, task)

    return result_dict


def get_reasoning_gym_options(answers, question, options, task):
    correct_group_text = answers
    wrong_group_text = list()
    for option in options:
        add = True
        for answ in answers:
            if answ == option:
                if not add:
                    print("WARNING: Answer found multiple times in option list!")
                add = False
        if add:
            wrong_group_text.append(option)

    return correct_group_text, wrong_group_text


# ------------------------------------------------------------------------------
# --------------------- reasoning gym specific code ----------------------------
# ------------------------------------------------------------------------------


def process_codeio(doc, results):
    """Custom processing function used to implement codeio metric from reasoning--gym."""

    # This class has the score function
    codeio_class = CodeIODataset(CodeIOConfig())

    # Assert we are evaluating a single target. This is a limitation of this
    # bAbI implementation
    assert len(results) == 1, "only single predictions are supported"

    # Get the data
    response = results[0]
    matches = re.findall(r"(?<!\\boxed)\{.*?\}|\[.*?\]", response, re.DOTALL)
    if len(matches) == 0:
        # Look for a boxed thing at the end...
        matches = re.findall(r"\\boxed\{(.*?)\}", response, re.DOTALL)
    if len(matches) > 0:
        extracted_json_string = matches[-1]
        cleaned_json_string = re.sub(r"\s*//.*", "", extracted_json_string)
        response = cleaned_json_string.replace("\n", "")
    else:
        response_out = ""
        m = re.search(r"([A-Za-z]+)[^A-Za-z]*\Z", response, flags=re.DOTALL)
        if m:
            word = m.group(1).lower()
            if word in ("true", "false", "null"):
                response_out = word
        if len(response_out) != 0:
            response = response_out
        else:
            print("Could not find a JSON-like structure in the generated string.")

    answer = doc["answer"].replace("\\", "")

    # Score
    sample_score = codeio_class.score_answer(answer, {"answer": response})

    # Compile and return
    results = {
        "exact_match": True if sample_score == 1.0 else False,
        "score_match": sample_score,
        "a-vert_correct_score": 0.0,
        "a-vert_wrong_score": 0.0,
        "a-vert_match": False,
        "a-vert_valid": False,
    }

    return results


def find_bracketed_blocks(s: str, ignore_command="\\boxed"):
    results = []  # list of (start_idx, end_idx, text)
    stack = []  # list of (opening_char, start_idx)
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch in "{[":
            # check if it's preceded by the ignore command
            cmd_len = len(ignore_command)
            if i >= cmd_len and s[i - cmd_len : i] == ignore_command:
                # skip this opening (treat as literal part of command)
                i += 1
                continue
            # push to stack: store bracket type and index of the opening char
            stack.append((ch, i))
        elif ch in "}]":
            if not stack:
                i += 1
                continue
            open_ch, start_idx = stack[-1]
            # matching pairs
            if (open_ch == "{" and ch == "}") or (open_ch == "[" and ch == "]"):
                stack.pop()
                end_idx = i
                block_text = s[start_idx : end_idx + 1]
                # depth AFTER popping
                depth_after = len(stack)
                results.append(
                    {
                        "start": start_idx,
                        "end": end_idx,
                        "text": block_text,
                        "outermost": depth_after == 0,
                    }
                )
            else:
                # unbalanced or mismatched bracket — treat as no-op (or handle as error)
                pass
        i += 1

    return results
