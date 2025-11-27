"""
Grouping methods for candidate group ranking.
Provides different strategies to aggregate distances/scores from multiple candidates.
"""

import numpy as np
from typing import Callable


# Available grouping methods
AVAILABLE_GROUPING_METHODS = [
    "max",
    "mean",
]

# Dynamic methods (with parameters)
DYNAMIC_GROUPING_PREFIXES = [
    "mean_top_k_",
]


def validate_grouping_method(grouping_method: str) -> bool:
    """
    Validate if a grouping method is supported.

    Args:
        grouping_method: Name of the grouping method to validate

    Returns:
        True if the method is valid, False otherwise
    """
    # Check if it's a direct match
    if grouping_method in AVAILABLE_GROUPING_METHODS:
        return True

    # Check if it matches a dynamic pattern
    for prefix in DYNAMIC_GROUPING_PREFIXES:
        if grouping_method.startswith(prefix):
            # Validate that the suffix is a valid integer
            try:
                suffix = grouping_method[len(prefix) :]
                int(suffix)
                return True
            except ValueError:
                return False

    return False


def get_grouping_function(grouping_method: str) -> Callable:
    """
    Get the grouping function for the specified method.

    Args:
        grouping_method: Name of the grouping method

    Returns:
        Callable function that takes an array and returns aggregated value

    Raises:
        ValueError: If the grouping method is not supported
    """
    if not validate_grouping_method(grouping_method):
        available = ", ".join(AVAILABLE_GROUPING_METHODS)
        dynamic = ", ".join([f"{p}<k>" for p in DYNAMIC_GROUPING_PREFIXES])
        raise ValueError(
            f"Grouping method '{grouping_method}' is not supported. "
            f"Available methods: {available}. "
            f"Dynamic methods: {dynamic}"
        )

    # Static methods
    if grouping_method == "max":
        return np.max
    elif grouping_method == "mean":
        return np.mean

    # Dynamic methods
    elif grouping_method.startswith("mean_top_k_"):
        top_k = int(grouping_method.split("mean_top_k_")[-1])
        return lambda x: np.mean(x[:top_k])

    # This should never be reached due to validation above
    raise ValueError(f"Unexpected grouping method: {grouping_method}")


def get_available_methods() -> list:
    """
    Get list of available grouping methods.

    Returns:
        List of available method names and patterns
    """
    methods = AVAILABLE_GROUPING_METHODS.copy()
    for prefix in DYNAMIC_GROUPING_PREFIXES:
        methods.append(f"{prefix}<k>")
    return methods
