"""
A-VERT: Augmented Verification and Retrieval Toolkit
"""

from a_vert.config import setup, get_available_templates, AvertConfig
from a_vert.grouping import get_available_methods as get_available_grouping_methods
from a_vert.logger import get_logger
from a_vert import processing
from a_vert import embedding_tools

__all__ = [
    "setup",
    "get_available_templates",
    "get_available_grouping_methods",
    "get_logger",
    "processing",
    "embedding_tools",
    "AvertConfig",
]
