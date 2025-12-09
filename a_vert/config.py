"""
Configuration module for A-VERT.
Handles environment variables and template configurations.
"""

import os
from typing import Dict, Any, Optional
import codecs

from a_vert import grouping
from a_vert.logger import get_logger

logger = get_logger(__name__)


class AvertConfig:
    """
    Configuration class for A-VERT.

    This class holds all configuration parameters needed for A-VERT processing.
    """

    def __init__(
        self,
        avert_method: str,
        document_template: Optional[str],
        query_template: Optional[str],
        grouping: str,
        enhance: bool,
        avert_model_endpoint: str,
        avert_endpoint_type: str,
        avert_model_name: Optional[str],
        instruction_map: Dict[str, str],
        instruction_flag: bool = False,
    ):
        """
        Initialize AvertConfig.

        Args:
            avert_method: Method to use ('embedding' or 'rerank')
            document_template: Template for document formatting (can be None)
            query_template: Template for query formatting (can be None)
            grouping: Grouping method to use
            enhance: Whether to enhance candidate groups
            avert_model_endpoint: Endpoint URL for the model
            avert_endpoint_type: Type of endpoint (e.g., 'vllm', 'openai')
            avert_model_name: Name of the model (can be None for some endpoints)
            instruction_map: Dictionary mapping task names to instructions
            instruction_flag: Whether instruction injection is enabled
        """
        self.avert_method = avert_method
        self.document_template = document_template
        self.query_template = query_template
        self.grouping = grouping
        self.enhance = enhance
        self.avert_model_endpoint = avert_model_endpoint
        self.avert_endpoint_type = avert_endpoint_type
        self.avert_model_name = avert_model_name
        self.instruction_map = instruction_map
        self.instruction_flag = instruction_flag

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AvertConfig":
        """
        Create AvertConfig from a dictionary (typically from setup()).

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            AvertConfig instance
        """
        return cls(
            avert_method=config_dict["AVERT_METHOD"],
            document_template=config_dict.get("DOCUMENT_TEMPLATE"),
            query_template=config_dict.get("QUERY_TEMPLATE"),
            grouping=config_dict["GROUPING"],
            enhance=config_dict["ENHANCE"],
            avert_model_endpoint=config_dict["AVERT_MODEL_ENDPOINT"],
            avert_endpoint_type=config_dict["AVERT_ENDPOINT_TYPE"],
            avert_model_name=config_dict.get("AVERT_MODEL_NAME"),
            instruction_map=config_dict.get("INSTRUCTION_MAP", {}),
            instruction_flag=config_dict.get("INSTRUCTION_FLAG", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AvertConfig to dictionary format.

        Returns:
            Dictionary with all configuration values
        """
        return {
            "AVERT_METHOD": self.avert_method,
            "DOCUMENT_TEMPLATE": self.document_template,
            "QUERY_TEMPLATE": self.query_template,
            "GROUPING": self.grouping,
            "ENHANCE": self.enhance,
            "AVERT_MODEL_ENDPOINT": self.avert_model_endpoint,
            "AVERT_ENDPOINT_TYPE": self.avert_endpoint_type,
            "AVERT_MODEL_NAME": self.avert_model_name,
            "INSTRUCTION_MAP": self.instruction_map,
            "INSTRUCTION_FLAG": self.instruction_flag,
        }


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


def setup(instruction_map={}) -> AvertConfig:
    """
    Setup and validate A-VERT configuration from environment variables.

    Returns:
        AvertConfig instance with all configuration parameters.

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

        # Both templates must be provided when using custom templates
        if custom_doc_template is None and custom_query_template is None:
            raise ValueError(
                "Either AVERT_PROMPT_TEMPLATE must be set to a predefined template name, "
                "or AVERT_DOCUMENT_TEMPLATE and AVERT_QUERY_TEMPLATE must be provided. "
                f"Available predefined templates: {', '.join(PREDEFINED_TEMPLATES.keys())}"
            )

        # If one custom template is provided, both must be provided
        if (custom_doc_template is None) != (custom_query_template is None):
            missing = (
                "AVERT_DOCUMENT_TEMPLATE"
                if custom_doc_template is None
                else "AVERT_QUERY_TEMPLATE"
            )
            raise ValueError(
                f"Both AVERT_DOCUMENT_TEMPLATE and AVERT_QUERY_TEMPLATE must be provided when using custom templates. "
                f"Missing: {missing}"
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

    # --- Instruction map loading & structural validation (no injection here) ---
    instruction_path = os.getenv("AVERT_INSTRUCTION_CONFIG_PATH")
    if instruction_path:
        try:
            import json

            with open(instruction_path, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            if not isinstance(loaded_map, dict):
                raise ValueError(
                    "Instruction config JSON must have a top-level object (mapping task names to strings)."
                )
            for k, v in loaded_map.items():
                if not isinstance(v, str):
                    raise ValueError(
                        f"Instruction config value for key '{k}' must be a string, got {type(v)}"
                    )
            instruction_map = loaded_map
        except FileNotFoundError:
            raise ValueError(
                f"AVERT_INSTRUCTION_CONFIG_PATH points to a missing file: {instruction_path}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load instruction config JSON: {e}")

    # Precedence: AVERT_INSTRUCTION_PROMPT environment variable overrides JSON "default" key
    # This allows quick testing/override without modifying the JSON file.
    override_prompt = os.getenv("AVERT_INSTRUCTION_PROMPT")
    if override_prompt is not None and override_prompt.strip() != "":
        # Override/create default entry
        instruction_map["default"] = override_prompt.strip()

    config["INSTRUCTION_MAP"] = instruction_map

    # Structural validation: '{instruction}' must appear in at most ONE template
    doc_template = config.get("DOCUMENT_TEMPLATE")
    query_template = config.get("QUERY_TEMPLATE")
    placeholder_in_doc = bool(doc_template and "{instruction}" in doc_template)
    placeholder_in_query = bool(query_template and "{instruction}" in query_template)
    if placeholder_in_doc and placeholder_in_query:
        raise ValueError(
            "'{instruction}' placeholder cannot appear in both document and query templates simultaneously."
        )
    # If any template uses '{instruction}', ensure a default instruction exists
    config["INSTRUCTION_FLAG"] = False
    if placeholder_in_doc or placeholder_in_query:
        config["INSTRUCTION_FLAG"] = True
        default_instr = instruction_map.get("default", None)
        # if no default instruction is None, log warning! and continue without raising error
        if default_instr is None:
            logger.warning(
                "Templates include '{instruction}' but no default instruction was provided. "
                "To avoid this warning and future errors, you can"
                " set `AVERT_INSTRUCTION_PROMPT`,"
                " define a 'default' entry in `AVERT_INSTRUCTION_CONFIG_PATH` .json file, or"
                " define a `INSTRUCTION_MAP` variable with a dictionary that includes a 'default' key with its corresponding instruction string."
            )
    # Return AvertConfig instance instead of dictionary
    return AvertConfig.from_dict(config)


def get_available_templates() -> list:
    """
    Get list of available predefined template names.

    Returns:
        List of template names.
    """
    return list(PREDEFINED_TEMPLATES.keys())
