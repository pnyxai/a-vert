# These represents instances where the LM refuses to answer due to not being confident or being censored.
refusal_group_text = [
    "I'm unable to answer that question.",
    "Sorry, I'm not allowed to answer that.",
    # "That question falls outside my scope.",
    "I choose not to respond to that inquiry.",
]

# These represent instances where the LM believes the questions is actually wrong and therefore cannot be correctly answered
formulation_mistake_base_group_text = [
    "The question cannot be answered since the problem statement seems to be flawed or inconsistent.",
    "There might be a mistake in the question formulation.",
]

formulation_mistake_choices_group_text = [
    "The correct answer is not listed in the available options.",
    "There is no valid answer among the choices given.",
    "The correct option is not listed among the given choices. If I must choose the closest is"
]

# Structures used to enhance answer groups
num2cardinal = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth"
    }
num2num = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "10"
    }
num2roman = {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
        8: "VIII",
        9: "IX",
        10: "X"
    }
num2letter = {
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E",
        6: "F",
        7: "G",
        8: "H",
        9: "I",
        10: "J"
    }
def get_symbol_dict(option_symbol):
    if option_symbol == "letters":
        num2symb = num2letter
    elif option_symbol == "cardinals":
        num2symb = num2cardinal
    elif option_symbol == "romans":
        num2symb = num2roman
    elif option_symbol == "numbers":
        num2symb = num2num
    else:
        raise ValueError(f"symbol option: \"{option_symbol}\" not supported.")
    return num2symb


# Function to enhance the target groups
def enhance_group(group_text, group_idxs, with_options=False, option_symbol="letters", return_references=False):
    """Given a list of candidate texts, produces a larger list based on the 
    provided texts and some common enhancements.
    """

    # Output list and their references (used for tracking and debug)
    out_list = list()
    reference_list = list()
    for response, idx in zip(group_text, group_idxs):
        # Add plain candidate to output list
        out_list += [response]
        # Create enhancements for this candidate
        postion_responses = [
            f"The answer is : \"{response}\" . Let me explain why",
            f"Therefore, the answer is : \"{response}\"",            
        ]
        # Track references
        if return_references:
            for i in range(len(postion_responses)):
                reference_list.append(f"enhancement_{i+1}")

        # If the candidates are part of a multiple-choice question, apply 
        # specific enhancements
        if with_options:
            # Get the symbols used to identify the options (A,B,.. 1,2,... I,II,III,...)
            num2symb = get_symbol_dict(option_symbol)
            symbol = num2symb[idx+1]
            # Add option candidates
            candidates = [
                f"Therefore, the correct answer is option \"{symbol}\": \"{response}\"",
                f"the answer is option \"{symbol}\": \"{response}\"",
            ]
            postion_responses += candidates
            # Track
            if return_references:
                for i in range(len(candidates)):
                    reference_list.append(f"enhancement_options_{i+1}")
        # Add all enhancements to output list
        out_list += postion_responses


    if return_references:
        return out_list, reference_list
    else:
        return out_list

def all_options_group(correct_text, correct_idx, wrong_texts, wrong_idxs, option_symbol="letters", return_references=False):
    """For multiple-choice questions, the enhancements can contain all other candidates. This function creates 
    candidates that mention other options but select a specific one.
    """
    # Get the symbol to be used
    num2symb = get_symbol_dict(option_symbol)
    # Get the correct (in this call) target
    correct_cardinal = num2cardinal[correct_idx+1]
    correct_symbol = num2symb[correct_idx+1]
    # Create the list of mentions to other (wrong) candidates
    wrongs = ""
    all_options = ""
    for response, idx in zip(wrong_texts, wrong_idxs):
        if correct_idx < idx and correct_idx == idx-1:
            all_options += f"\tOption \"{correct_symbol}\". \"{correct_text}\". Is correct.\n"
        symbol = num2symb[idx+1]
        this = f"\tOption \"{symbol}\". \"{response}\". Is not correct.\n"
        wrongs += this
        all_options += this
    # Fill the candidate list, mentioning each of the choices in the answer
    candidate_list = [
        f"The answer is the {correct_cardinal} one, option \"{correct_symbol}\". \"{correct_text}\". Let me explain why:\n"+wrongs, 
        f"Analyzing the options:\n"+all_options+f"\nTherefore, the answer is the {correct_cardinal} one, option \"{correct_symbol}\". \"{correct_text}\""
        ]

    if return_references:
        return candidate_list, [f"enhancement_options_groups_{i+1}" for i in range(len(candidate_list))]
    else:
        return candidate_list

# TODO
# def add_negation_group()