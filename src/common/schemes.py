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
    evidence: str = Field(..., max_length=500, description="Short snippet from original text")


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
    evidence: str = Field(..., max_length=500, description="Short snippet from original text")

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

# ============================================================
# INTEGRATED RESULTS OUTPUT MODELS
# ============================================================

class EvidenceType(str, Enum):
    """Classification of evidence quality"""
    NO_EVIDENCE = "no_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"
    CONSISTENT_SOURCES = "consistent_sources"

class RiskLevel(str, Enum):
    """Risk level for investment decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ClaimCategory(str, Enum):
    """Auto-categorization of claim types"""
    FINANCIAL = "financial"
    MARKET_POSITION = "market_position"
    TECHNOLOGY = "technology"
    PARTNERSHIPS = "partnerships"
    TEAM = "team"
    PRODUCT = "product"
    REGULATORY = "regulatory"
    OTHER = "other"

class EnhancedVerification(BaseModel):
    """Enhanced verification result with evidence type classification"""
    verdict: Verdict = Field(description="SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation for the verdict")
    evidence_type: EvidenceType = Field(description="Type of evidence found")
    source_count: int = Field(ge=0, description="Number of sources found")
    sources: List[str] = Field(default_factory=list, description="List of source URLs")
    contradiction_details: str | None = Field(None, description="Details when sources conflict")

class RiskAssessment(BaseModel):
    """Investment risk assessment for a claim"""
    risk_level: RiskLevel = Field(description="Overall risk level")
    investor_alert: bool = Field(description="Flag for critical issues")
    requires_followup: bool = Field(description="Whether additional verification needed")
    followup_questions: List[str] = Field(default_factory=list, description="Specific questions for due diligence")

class IntegratedClaimResult(BaseModel):
    """Single claim with complete verification and risk assessment"""
    claim_id: str = Field(description="Unique claim identifier")
    claim_text: str = Field(description="Plain English description of the claim")
    category: ClaimCategory = Field(description="Auto-categorized claim type")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    original_evidence: str = Field(description="Original evidence snippet from document")
    verification: EnhancedVerification = Field(description="Verification details")
    risk_assessment: RiskAssessment = Field(description="Risk assessment")

class ResultSummary(BaseModel):
    """Summary statistics for all claims"""
    total_claims: int = Field(ge=0, description="Total number of claims")
    supported: int = Field(ge=0, description="Number of supported claims")
    contradicted: int = Field(ge=0, description="Number of contradicted claims")
    insufficient_evidence: int = Field(ge=0, description="Number of claims with insufficient evidence")
    high_risk_count: int = Field(ge=0, description="Number of high-risk claims")
    evidence_breakdown: dict[str, int] = Field(description="Count by evidence type")

class IntegratedReport(BaseModel):
    """Complete integrated report with all claims and verification"""
    company_name: str = Field(description="Name of the company being analyzed")
    processed_at: str = Field(description="ISO timestamp of processing")
    summary: ResultSummary = Field(description="Summary statistics")
    results: List[IntegratedClaimResult] = Field(default_factory=list, description="All claim results")
    
    def to_dict(self) -> dict:
        return self.model_dump()
    
    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)