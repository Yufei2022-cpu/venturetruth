"""
Data schemas for claim extraction.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class Claim(BaseModel):
    """Individual claim model."""
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
        validate_assignment=True
    )

    id: str = Field(..., description="Unique claim identifier (e.g., C1, C2)")
    claim: str = Field(..., description="Plain English description of the claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    evidence: str = Field(..., max_length=200, description="Short snippet from original text")


class ClaimsResponse(BaseModel):
    """Response model containing all extracted claims."""
    model_config = ConfigDict(extra='forbid')

    claims: List[Claim] = Field(default_factory=list, description="List of extracted claims")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(**kwargs)