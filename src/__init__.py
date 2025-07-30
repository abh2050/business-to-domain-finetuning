"""
AI Engineer Homework: Domain Name Suggestion LLM
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "AI Engineer Candidate"
__email__ = "candidate@example.com"
__description__ = "Fine-tuned LLM for domain name suggestions with systematic evaluation"

from .data_generation import SyntheticDataGenerator
from .model import DomainNameModel
from .evaluation import LLMJudgeEvaluator
from .safety import SafetyFilter

__all__ = [
    "SyntheticDataGenerator",
    "DomainNameModel", 
    "LLMJudgeEvaluator",
    "SafetyFilter"
]
