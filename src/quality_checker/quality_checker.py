"""
Quality Checker for Claim Verification Results
Uses GPT-5.2-Thinking to analyze and critique verification quality
"""

import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

load_dotenv()


class QualityRating(str, Enum):
    """Overall quality rating for a verification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class IssueType(str, Enum):
    """Types of quality issues found"""
    REASONING_GAP = "reasoning_gap"              # Verdict doesn't follow from reasoning
    SOURCE_MISMATCH = "source_mismatch"          # Sources don't support claims made
    CONFIDENCE_MISCALIBRATION = "confidence_miscalibration"  # Confidence too high/low
    MISSING_VERIFICATION = "missing_verification"  # Claim not properly verified
    CONTRADICTORY_LOGIC = "contradictory_logic"   # Internal contradictions
    INSUFFICIENT_DEPTH = "insufficient_depth"     # Analysis too shallow
    HALLUCINATED_FACTS = "hallucinated_facts"     # Made up information
    CATEGORY_ERROR = "category_error"             # Wrong claim category


class QualityIssue(BaseModel):
    """Individual quality issue found"""
    claim_id: str = Field(description="ID of the claim with the issue")
    issue_type: IssueType = Field(description="Type of quality issue")
    severity: str = Field(description="high, medium, or low")
    description: str = Field(description="Detailed description of the issue")
    recommendation: str = Field(description="How to fix this issue")


class ClaimQualityAssessment(BaseModel):
    """Quality assessment for a single claim verification"""
    claim_id: str = Field(description="ID of the claim being assessed")
    quality_rating: QualityRating = Field(description="Overall quality rating")
    reasoning_quality: float = Field(ge=0.0, le=1.0, description="How well reasoning supports verdict")
    source_relevance: float = Field(ge=0.0, le=1.0, description="How relevant sources are")
    certainty_calibration: float = Field(ge=0.0, le=1.0, description="How well calibrated certainty is")
    issues: List[QualityIssue] = Field(default_factory=list, description="Issues found")
    strengths: List[str] = Field(default_factory=list, description="What was done well")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class QualityReport(BaseModel):
    """Complete quality assessment report - used for LLM structured output"""
    assessed_at: str = Field(description="ISO timestamp of assessment")
    company_name: str = Field(description="Company being assessed")
    overall_quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality 0-1")
    overall_rating: QualityRating = Field(description="Overall quality rating")
    total_claims_assessed: int = Field(description="Number of claims assessed")
    critical_issues_count: int = Field(description="Number of critical issues found")
    
    # Summary statistics
    excellent_count: int = Field(ge=0, description="Claims with excellent quality")
    good_count: int = Field(ge=0, description="Claims with good quality")
    acceptable_count: int = Field(ge=0, description="Claims with acceptable quality")
    poor_count: int = Field(ge=0, description="Claims with poor quality")
    critical_count: int = Field(ge=0, description="Claims with critical issues")
    
    # Detailed assessments
    claim_assessments: List[ClaimQualityAssessment] = Field(description="Per-claim assessments")
    
    # Overall recommendations
    top_issues: List[str] = Field(description="Most important issues to address")
    systemic_problems: List[str] = Field(description="Patterns of issues across claims")
    recommended_actions: List[str] = Field(description="Prioritized action items")
    
    def to_dict(self) -> dict:
        return self.model_dump()
    
    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)


class QualityChecker:
    """
    Analyzes the quality of claim verification results using GPT-5.2-Thinking
    """
    
    def __init__(self, api_key: str, model: str = "gpt-5.2-pro", temperature: float = 0.1):
        """
        Initialize the Quality Checker
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5.2-thinking for deep analysis)
            temperature: Model temperature (default: 0.1 for consistency)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.is_setup = False
        
    def setup(self):
        """Setup the quality checking chain"""
        chat_model = ChatOpenAI(
            model=self.model, 
            temperature=self.temperature, 
            api_key=self.api_key
        )
        structured_llm = chat_model.with_structured_output(QualityReport)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{report_data}")
        ])
        
        self.quality_chain = prompt | structured_llm
        self.is_setup = True
        
    def _get_system_prompt(self) -> str:
        return """You are an expert quality assurance analyst specializing in fact-checking and claim verification systems.

Your role is to critically evaluate the quality of claim verification results, identifying:
- Logical inconsistencies between verdicts and reasoning
- Misuse or misattribution of sources
- Confidence scores that don't match evidence quality
- Missing or incomplete verification steps
- Potential hallucinations or fabricated information
- Systematic biases or errors

EVALUATION CRITERIA:

1. REASONING QUALITY (0-1):
   - Does the reasoning logically support the verdict?
   - Are there gaps in the argument?
   - Is the analysis thorough enough?

2. SOURCE RELEVANCE (0-1):
   - Do the cited sources actually support the claims made?
   - Are sources credible and appropriate?
   - Were key sources missed?

3. CERTAINTY CALIBRATION (0-1):
   - Is the certainty level appropriate given the evidence?
   - High certainty should require strong evidence
   - Uncertain claims should have lower certainty

QUALITY RATINGS:
- EXCELLENT: Rigorous analysis, well-sourced, properly calibrated
- GOOD: Solid analysis with minor issues
- ACCEPTABLE: Adequate but could be improved
- POOR: Significant issues that could mislead
- CRITICAL: Fundamental problems requiring immediate attention

Be thorough but fair. Acknowledge strengths while highlighting areas for improvement.
Focus on actionable feedback that can improve the verification pipeline."""

    def _build_analysis_prompt(self, report: dict, search_data: dict = None) -> str:
        """Build the analysis prompt from report data"""
        search_section = ""
        if search_data and search_data.get("search_results"):
            search_section = f"""

SEARCH RESULTS DATA (for analyzing search quality):
{json.dumps(search_data["search_results"], indent=2, ensure_ascii=False)}

When analyzing, also evaluate:
- Were search queries targeted and specific?
- Were the right entities identified?
- What percentage of sources were relevant vs off-topic?
- Did failed searches impact verification quality?
"""
        
        return f"""TASK: Analyze the quality of this claim verification report.

COMPANY: {report.get('company_name', 'Unknown')}

REPORT DATA:
{json.dumps(report, indent=2, ensure_ascii=False)}
{search_section}
ANALYSIS INSTRUCTIONS:

1. For each claim verification result:
   - Evaluate if the verdict is justified by the reasoning
   - Check if sources support the conclusions drawn
   - Assess if certainty level is appropriate
   - Identify any logical gaps or issues
   - Note what was done well

2. Look for systemic issues across all claims:
   - Patterns of over/under-certainty
   - Common reasoning gaps
   - Source quality issues
   - Category misclassifications
   - Search retrieval problems (if search data provided)

3. Provide actionable recommendations:
   - Priority-ordered action items
   - Specific fixes for critical issues
   - Process improvements

Be critical but constructive. Your goal is to improve verification quality."""

    def check_quality(self, report_path: str, search_results: list = None) -> QualityReport:
        """
        Analyze the quality of a verification report
        
        Args:
            report_path: Path to the final_report.json file
            search_results: Optional list of search results per company
            
        Returns:
            QualityReport with detailed quality assessment
        """
        if not self.is_setup:
            raise RuntimeError("Please call setup() before checking quality")
            
        # Load the report
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        # Calculate search quality summary if search results provided
        search_quality_summary = None
        if search_results:
            search_quality_summary = self._analyze_search_quality(search_results)
            
        # Process each company in the report
        all_assessments = []
        
        companies = report_data.get("companies", [report_data])
        
        for idx, company_report in enumerate(companies):
            # Add search results to the analysis if available
            company_search = None
            if search_results and idx < len(search_results):
                company_search = search_results[idx]
                
            analysis_prompt = self._build_analysis_prompt(company_report, company_search)
            
            quality_report = self.quality_chain.invoke({
                "report_data": analysis_prompt
            })
            
            all_assessments.append(quality_report)
            
        # If multiple companies, aggregate
        if len(all_assessments) == 1:
            result = all_assessments[0]
        else:
            result = self._aggregate_reports(all_assessments)
            
        # Return both the report and search quality summary
        return result, search_quality_summary
    
    def _analyze_search_quality(self, search_results: list) -> dict:
        """Analyze overall search quality across all companies"""
        total_searches = 0
        excellent_count = 0
        good_count = 0
        partial_count = 0
        failed_count = 0
        off_topic_sources = 0
        total_sources = 0
        
        per_claim_details = []
        
        for company_data in search_results:
            if not company_data or not company_data.get("search_results"):
                continue
                
            search_list = company_data["search_results"].get("search_results_list", [])
            
            for search in search_list:
                total_searches += 1
                quality = search.get("search_quality", "UNKNOWN")
                
                if quality == "EXCELLENT":
                    excellent_count += 1
                elif quality == "GOOD":
                    good_count += 1
                elif quality == "PARTIAL":
                    partial_count += 1
                elif quality == "FAILED":
                    failed_count += 1
                    
                # Count source relevance
                source_details = search.get("source_details", [])
                for source in source_details:
                    total_sources += 1
                    if source.get("relevance") == "OFF_TOPIC":
                        off_topic_sources += 1
                        
                # Collect per-claim details
                claim = search.get("claim", {})
                per_claim_details.append({
                    "claim_id": claim.get("id", "unknown"),
                    "search_query": search.get("search_query", ""),
                    "entity_identified": search.get("entity_identified", ""),
                    "search_quality": quality,
                    "relevant_sources": len([s for s in source_details if s.get("relevance") in ["HIGH", "MEDIUM"]]),
                    "off_topic_sources": len([s for s in source_details if s.get("relevance") == "OFF_TOPIC"]),
                    "search_notes": search.get("search_notes")
                })
        
        # Calculate additional metrics
        high_relevance_sources = sum(
            len([s for s in search.get("source_details", []) if s.get("relevance") == "HIGH"])
            for company_data in search_results
            if company_data and company_data.get("search_results")
            for search in company_data["search_results"].get("search_results_list", [])
        )
        
        total_retries = sum(
            search.get("retry_count", 0)
            for company_data in search_results
            if company_data and company_data.get("search_results")
            for search in company_data["search_results"].get("search_results_list", [])
        )
        
        total_domains = sum(
            search.get("unique_domains", 0)
            for company_data in search_results
            if company_data and company_data.get("search_results")
            for search in company_data["search_results"].get("search_results_list", [])
        )
        
        return {
            "total_searches": total_searches,
            "quality_breakdown": {
                "EXCELLENT": excellent_count,
                "GOOD": good_count,
                "PARTIAL": partial_count,
                "FAILED": failed_count
            },
            "excellent_rate": round(excellent_count / total_searches, 2) if total_searches > 0 else 0,
            "good_rate": round((excellent_count + good_count) / total_searches, 2) if total_searches > 0 else 0,
            "failed_rate": round(failed_count / total_searches, 2) if total_searches > 0 else 0,
            "total_sources": total_sources,
            "off_topic_sources": off_topic_sources,
            "off_topic_rate": round(off_topic_sources / total_sources, 2) if total_sources > 0 else 0,
            # New metrics
            "avg_sources_per_claim": round(total_sources / total_searches, 1) if total_searches > 0 else 0,
            "high_relevance_rate": round(high_relevance_sources / total_sources, 2) if total_sources > 0 else 0,
            "total_retries": total_retries,
            "avg_domains_per_claim": round(total_domains / total_searches, 1) if total_searches > 0 else 0,
            "per_claim_search_details": per_claim_details
        }
    
    def check_quality_from_dict(self, report_data: dict) -> QualityReport:
        """
        Analyze quality from a dictionary (in-memory report)
        
        Args:
            report_data: Report data as dictionary
            
        Returns:
            QualityReport with detailed quality assessment
        """
        if not self.is_setup:
            raise RuntimeError("Please call setup() before checking quality")
            
        analysis_prompt = self._build_analysis_prompt(report_data)
        
        return self.quality_chain.invoke({
            "report_data": analysis_prompt
        })
    
    def _aggregate_reports(self, reports: List[QualityReport]) -> QualityReport:
        """Aggregate multiple quality reports into one"""
        total_claims = sum(r.total_claims_assessed for r in reports)
        avg_score = sum(r.overall_quality_score for r in reports) / len(reports)
        
        all_issues = []
        all_systemic = []
        all_actions = []
        all_assessments = []
        
        for r in reports:
            all_issues.extend(r.top_issues)
            all_systemic.extend(r.systemic_problems)
            all_actions.extend(r.recommended_actions)
            all_assessments.extend(r.claim_assessments)
            
        # Determine overall rating from average score
        if avg_score >= 0.9:
            overall_rating = QualityRating.EXCELLENT
        elif avg_score >= 0.75:
            overall_rating = QualityRating.GOOD
        elif avg_score >= 0.6:
            overall_rating = QualityRating.ACCEPTABLE
        elif avg_score >= 0.4:
            overall_rating = QualityRating.POOR
        else:
            overall_rating = QualityRating.CRITICAL
            
        return QualityReport(
            assessed_at=datetime.now().isoformat(),
            company_name="Multi-Company Aggregate",
            overall_quality_score=avg_score,
            overall_rating=overall_rating,
            total_claims_assessed=total_claims,
            critical_issues_count=sum(r.critical_issues_count for r in reports),
            excellent_count=sum(r.excellent_count for r in reports),
            good_count=sum(r.good_count for r in reports),
            acceptable_count=sum(r.acceptable_count for r in reports),
            poor_count=sum(r.poor_count for r in reports),
            critical_count=sum(r.critical_count for r in reports),
            claim_assessments=all_assessments,
            top_issues=list(set(all_issues))[:10],
            systemic_problems=list(set(all_systemic)),
            recommended_actions=list(set(all_actions))[:10]
        )
    
    def store_as_json(self, quality_report: QualityReport, path: str = "res/quality_report.json", search_quality_summary: dict = None):
        """Save quality report to JSON file, optionally including search quality summary"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dict and add search quality summary if provided
        output_data = quality_report.to_dict()
        if search_quality_summary:
            output_data["search_quality_summary"] = search_quality_summary
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"Quality report saved to: {path}")


def main():
    """Example usage of Quality Checker"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize checker with GPT-5.2-Thinking
    checker = QualityChecker(api_key=api_key, model="gpt-5.2-thinking")
    checker.setup()
    
    # Check quality of final report
    project_root = Path(__file__).parent.parent.parent
    report_path = project_root / "res" / "final_report.json"
    
    print("🔍 Analyzing verification quality...")
    quality_report = checker.check_quality(str(report_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 QUALITY ASSESSMENT RESULTS")
    print("=" * 60)
    print(f"Overall Score: {quality_report.overall_quality_score:.2f}")
    print(f"Overall Rating: {quality_report.overall_rating.value.upper()}")
    print(f"Total Claims Assessed: {quality_report.total_claims_assessed}")
    print(f"Critical Issues: {quality_report.critical_issues_count}")
    
    print("\n📈 Quality Distribution:")
    print(f"   ✨ Excellent: {quality_report.excellent_count}")
    print(f"   ✅ Good: {quality_report.good_count}")
    print(f"   ⚠️  Acceptable: {quality_report.acceptable_count}")
    print(f"   ❌ Poor: {quality_report.poor_count}")
    print(f"   🚨 Critical: {quality_report.critical_count}")
    
    if quality_report.top_issues:
        print("\n🔴 Top Issues:")
        for i, issue in enumerate(quality_report.top_issues[:5], 1):
            print(f"   {i}. {issue}")
    
    if quality_report.recommended_actions:
        print("\n💡 Recommended Actions:")
        for i, action in enumerate(quality_report.recommended_actions[:5], 1):
            print(f"   {i}. {action}")
    
    # Save detailed report
    output_path = project_root / "res" / "quality_report.json"
    checker.store_as_json(quality_report, str(output_path))


if __name__ == "__main__":
    main()
