"""
Claim Extractor - Extract factual claims from text using OpenAI API.
"""

from .core import extract_claims
from .schemas import Claim, ClaimsResponse

__version__ = "1.0.0"
__all__ = ["extract_claims", "Claim", "ClaimsResponse"]