from enum import Enum
from typing import List
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
    
class SearchResults(BaseModel):
    claim: Claim = Field(description="Claim for which the search results are provided")
    search_results: str = Field(description="Results of the internet search")
    sources: list[str] = Field(description="Sources for the provided search results")
    
class SearchResultsList(BaseModel):
    search_results_list: list[SearchResults] = Field(description="List of all search results")

class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    
class VerificationResult(BaseModel):
    claim_id: str = Field(description="The ID of the claim that is verified")
    verdict: Verdict = Field(description="Specifies whether the claim is SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE")
    confidence: float = Field(description="Number specifies how confident you are the verdict. The value ranges from 0 to 1 with at most 2 digits after the coma.")
    reasoning: str = Field(description="Explanation for your verdict. 2-4 sentences")
    sources: list[str] = Field(description="List of the URL sources used to support the reasoning")
    
class VerificationList(BaseModel):
    verification_results: list[VerificationResult] = Field(description="List of the verification results")

class Evidence(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
        validate_assignment=True
    )

    id: str = Field(..., description="Unique evidence identifier (e.g., E1, E2)")
    file_name: str = Field(..., description="File name of the evidence document")
    page_number: int = Field(..., description="Page number of the evidence document")
    evidence: str = Field(..., max_length=200, description="Short snippet from original text")

class ClaimEvidence(BaseModel):
    id: str = Field(..., description="Unique claim identifier (e.g., C1, C2)")
    claim: str = Field(..., description="Claim for which the evidence is provided")
    evidence: Evidence = Field(description="Evidence for the provided claim")

class FilteredClaims(BaseModel):
    model_config = ConfigDict(extra='forbid')

    claims: List[ClaimEvidence] = Field(default_factory=list, description="List of filtered claims")

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)