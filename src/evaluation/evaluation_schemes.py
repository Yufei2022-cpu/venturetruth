from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class FieldMetrics(BaseModel):
    """Metrics for a single metadata field."""
    field_name: str = Field(description="Name of the metadata field")
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(ge=0.0, le=1.0, description="F1 score")
    accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy score")
    true_positives: int = Field(ge=0, description="Number of correct matches")
    false_positives: int = Field(ge=0, description="Number of incorrect extractions")
    false_negatives: int = Field(ge=0, description="Number of missed values")
    total_samples: int = Field(ge=0, description="Total number of samples evaluated")


class CompanyEvaluationResult(BaseModel):
    """Evaluation result for a single company."""
    company_identifier: str = Field(description="Company identifier used for matching")
    matched: bool = Field(description="Whether ground truth was found for this company")
    field_results: Dict[str, dict] = Field(
        default_factory=dict,
        description="Per-field comparison: {field: {extracted, expected, match}}"
    )
    accuracy: float = Field(ge=0.0, le=1.0, description="Overall accuracy for this company")


class MetadataEvaluationReport(BaseModel):
    """Complete evaluation report across all companies."""
    evaluated_at: str = Field(description="ISO timestamp of evaluation")
    total_companies: int = Field(ge=0, description="Total companies evaluated")
    matched_companies: int = Field(ge=0, description="Companies with ground truth found")
    
    # Overall metrics
    overall_precision: float = Field(ge=0.0, le=1.0, description="Overall precision")
    overall_recall: float = Field(ge=0.0, le=1.0, description="Overall recall")
    overall_f1: float = Field(ge=0.0, le=1.0, description="Overall F1 score")
    overall_accuracy: float = Field(ge=0.0, le=1.0, description="Overall accuracy")
    
    # Per-field breakdown
    field_metrics: List[FieldMetrics] = Field(
        default_factory=list,
        description="Metrics broken down by field"
    )
    
    # Per-company results
    company_results: List[CompanyEvaluationResult] = Field(
        default_factory=list,
        description="Individual company evaluation results"
    )
    
    # Confusion matrix for categorical fields
    confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = Field(
        default_factory=dict,
        description="Confusion matrices for categorical fields like country/sector"
    )
    
    def to_dict(self) -> dict:
        return self.model_dump()
    
    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)
