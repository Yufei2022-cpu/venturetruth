from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ClaimEvidenceStats(BaseModel):
    """Evidence statistics for a single claim."""
    claim_id: str = Field(description="Claim identifier")
    claim_text: str = Field(description="The claim text")
    verdict: str = Field(description="Verification verdict")
    certainty: float = Field(ge=0.0, le=1.0, description="Certainty score")
    source_count: int = Field(ge=0, description="Number of sources used")
    unique_domains: int = Field(ge=0, description="Number of unique domains")
    source_types: Dict[str, int] = Field(default_factory=dict, description="Source types breakdown")
    has_sufficient_evidence: bool = Field(description="Whether evidence threshold met")


class EvidenceAnalysisReport(BaseModel):
    """Complete evidence analysis report."""
    analyzed_at: str = Field(description="ISO timestamp")
    total_claims: int = Field(ge=0)
    
    # Aggregate stats
    avg_sources_per_claim: float = Field(ge=0.0)
    avg_sources_supported: float = Field(ge=0.0, description="Avg sources for SUPPORTED claims")
    avg_sources_contradicted: float = Field(ge=0.0, description="Avg sources for CONTRADICTED claims")
    avg_sources_insufficient: float = Field(ge=0.0, description="Avg sources for INSUFFICIENT_EVIDENCE")
    
    # Source distribution
    claims_with_no_sources: int = Field(ge=0)
    claims_with_single_source: int = Field(ge=0)
    claims_with_multiple_sources: int = Field(ge=0)
    
    # Verdict breakdown by evidence strength
    evidence_verdict_correlation: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Correlation: source_count_bucket -> {verdict -> count}"
    )
    
    # Per-claim details
    claim_details: List[ClaimEvidenceStats] = Field(default_factory=list)
    
    # Recommendations
    low_evidence_claims: List[str] = Field(
        default_factory=list,
        description="Claims with insufficient sources for their verdict"
    )
    
    def to_dict(self) -> dict:
        return self.model_dump()


class RobustnessResult(BaseModel):
    """Result of robustness check for a single claim."""
    claim_id: str = Field(description="Claim identifier")
    claim_text: str = Field(description="The claim text")
    verdicts_across_runs: List[str] = Field(description="Verdicts from each run")
    certainties_across_runs: List[float] = Field(description="Certainties from each run")
    is_stable: bool = Field(description="Whether verdict was consistent across runs")
    consistency_rate: float = Field(ge=0.0, le=1.0, description="% of runs with same verdict")
    majority_verdict: str = Field(description="Most common verdict")
    certainty_variance: float = Field(ge=0.0, description="Variance in certainty scores")


class RobustnessReport(BaseModel):
    """Complete robustness analysis report."""
    analyzed_at: str = Field(description="ISO timestamp")
    num_runs: int = Field(description="Number of verification runs")
    total_claims: int = Field(ge=0)
    
    # Overall stability metrics
    overall_stability_rate: float = Field(
        ge=0.0, le=1.0, 
        description="% of claims with consistent verdict"
    )
    avg_certainty_variance: float = Field(ge=0.0)
    
    # Stability by verdict
    stability_by_verdict: Dict[str, float] = Field(
        default_factory=dict,
        description="Stability rate per verdict type"
    )
    
    # Unstable claims to investigate
    unstable_claims: List[RobustnessResult] = Field(
        default_factory=list,
        description="Claims with inconsistent verdicts"
    )
    
    # All claim results
    claim_results: List[RobustnessResult] = Field(default_factory=list)
    
    def to_dict(self) -> dict:
        return self.model_dump()
