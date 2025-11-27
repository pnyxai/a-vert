"""
Configuration module for A-VERT.
Handles environment variables and template configurations.
"""

import os
from typing import Dict, Any
import codecs

from a_vert import grouping

# Predefined templates mapping
PREDEFINED_TEMPLATES = {
    "qwen3-reranker": {
        "document_template": "<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "query_template": """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n <Instruct>: {instruction}\n<Query>: {query}\n""",
    },
    "empty": {
        "document_template": None,
        "query_template": None,
    },
    "embedding-with-instruction": {
        "document_template": None,
        "query_template": "{instruction}{query}",
    },
}


def setup() -> Dict[str, Any]:
    """
    Setup and validate A-VERT configuration from environment variables.

    Returns:
        Dict containing all A-VERT configuration parameters.

    Raises:
        ValueError: If required environment variables are not set.
    """
    config = {}

    # --- A-VERT Model Configuration ---
    avert_endpoint = os.getenv("AVERT_MODEL_ENDPOINT", None)
    if avert_endpoint is None:
        raise ValueError(
            "AVERT_MODEL_ENDPOINT environment variable is not set. "
            "This is required for A-VERT to function."
        )
    config["AVERT_MODEL_ENDPOINT"] = avert_endpoint

    avert_endpoint_type = os.getenv("AVERT_ENDPOINT_TYPE", None)
    if avert_endpoint_type is None:
        raise ValueError(
            "AVERT_ENDPOINT_TYPE environment variable is not set. "
            "This is required for A-VERT to function."
        )
    config["AVERT_ENDPOINT_TYPE"] = avert_endpoint_type

    avert_model_name = os.getenv("AVERT_MODEL_NAME", None)
    if avert_model_name is None and avert_endpoint_type in ("vllm", "openai"):
        raise ValueError(
            "AVERT_MODEL_NAME environment variable is not set. "
            "This is required for vLLM or OpenAI endpoint to function."
        )
    config["AVERT_MODEL_NAME"] = avert_model_name

    # --- Template Configuration ---
    # Check for predefined template first
    template_name = os.getenv("AVERT_PROMPT_TEMPLATE", None)

    if template_name is not None:
        # User selected a predefined template
        if template_name not in PREDEFINED_TEMPLATES:
            available = ", ".join(PREDEFINED_TEMPLATES.keys())
            raise ValueError(
                f"Unknown AVERT_PROMPT_TEMPLATE: '{template_name}'. "
                f"Available options: {available}"
            )

        template_config = PREDEFINED_TEMPLATES[template_name]
        config["DOCUMENT_TEMPLATE"] = template_config["document_template"]
        config["QUERY_TEMPLATE"] = template_config["query_template"]
    else:
        # User must provide custom templates
        custom_doc_template = os.getenv("AVERT_DOCUMENT_TEMPLATE", None)
        custom_query_template = os.getenv("AVERT_QUERY_TEMPLATE", None)

        # Decode escape sequences to handle newlines etc. from env vars
        if custom_doc_template:
            custom_doc_template = codecs.decode(custom_doc_template, "unicode_escape")
        if custom_query_template:
            custom_query_template = codecs.decode(
                custom_query_template, "unicode_escape"
            )

        # At least one template should be provided if no predefined template is selected
        if custom_doc_template is None and custom_query_template is None:
            raise ValueError(
                "Either AVERT_PROMPT_TEMPLATE must be set to a predefined template name, "
                "or AVERT_DOCUMENT_TEMPLATE and AVERT_QUERY_TEMPLATE must be provided. "
                f"Available predefined templates: {', '.join(PREDEFINED_TEMPLATES.keys())}"
            )

        config["DOCUMENT_TEMPLATE"] = custom_doc_template
        config["QUERY_TEMPLATE"] = custom_query_template

    # --- Method Configuration ---
    # Method must always be provided by the user
    avert_method = os.getenv("AVERT_METHOD", None)
    if avert_method is None:
        raise ValueError(
            "AVERT_METHOD environment variable is not set. "
            "This is required for A-VERT to function. "
            "Must be either 'rerank' or 'embedding'."
        )
    if avert_method not in ("rerank", "embedding"):
        raise ValueError(
            f"Invalid AVERT_METHOD value: '{avert_method}'. "
            "Must be either 'rerank' or 'embedding'."
        )
    config["AVERT_METHOD"] = avert_method

    # --- GROUPING and ENHANCE Configuration ---
    # These are separate from templates and must be configured independently
    grouping_method = os.getenv("AVERT_GROUPING", "max")
    if not grouping.validate_grouping_method(grouping_method):
        available = ", ".join(grouping.get_available_methods())
        raise ValueError(
            f"Invalid AVERT_GROUPING value: '{grouping_method}'. "
            f"Available methods: {available}"
        )
    config["GROUPING"] = grouping_method

    enhance_str = os.getenv("AVERT_ENHANCE", "true").lower()
    if enhance_str not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(
            f"Invalid AVERT_ENHANCE value: '{enhance_str}'. "
            "Must be one of: true, false, 1, 0, yes, no"
        )
    config["ENHANCE"] = enhance_str in ("true", "1", "yes")

    # --- Instruction prompt validation and injection ---
    doc_template = config.get("DOCUMENT_TEMPLATE")
    query_template = config.get("QUERY_TEMPLATE")
    avert_instruction_prompt = os.getenv("AVERT_INSTRUCTION_PROMPT", "").strip()

    instruction_in_doc = bool(doc_template and "{instruction}" in doc_template)
    instruction_in_query = bool(query_template and "{instruction}" in query_template)

    if avert_instruction_prompt:
        # Placeholder must be in one template, but not both.
        if instruction_in_doc and instruction_in_query:
            raise ValueError(
                "The '{instruction}' placeholder cannot be present in both the "
                "document and query templates when AVERT_INSTRUCTION_PROMPT is set."
            )

        if not instruction_in_doc and not instruction_in_query:
            raise ValueError(
                "AVERT_INSTRUCTION_PROMPT is set, but the '{instruction}' placeholder "
                "was not found in the document or query template."
            )

        # Inject instruction
        if instruction_in_doc:
            config["DOCUMENT_TEMPLATE"] = doc_template.replace(
                "{instruction}", avert_instruction_prompt
            )
        if instruction_in_query:
            config["QUERY_TEMPLATE"] = query_template.replace(
                "{instruction}", avert_instruction_prompt
            )

    elif instruction_in_doc or instruction_in_query:
        location = "document template" if instruction_in_doc else "query template"
        raise ValueError(
            f"Found '{{instruction}}' in the {location}, but AVERT_INSTRUCTION_PROMPT environment variable is not set. Please set it to fill the placeholder."
        )

    return config


def get_available_templates() -> list:
    """
    Get list of available predefined template names.

    Returns:
        List of template names.
    """
    return list(PREDEFINED_TEMPLATES.keys())
