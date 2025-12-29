import os
import json
from datetime import datetime
from typing import List

from common.schemes import (
    ClaimsResponse, VerificationList, IntegratedReport, IntegratedClaimResult,
    EnhancedVerification, RiskAssessment, ResultSummary, EvidenceType, RiskLevel,
    ClaimCategory, Verdict
)


class ResultAggregator:
    """Aggregates claims and verification results into integrated report"""
    
    def __init__(self, company_name: str = "Unknown Company"):
        self.company_name = company_name
    
    def aggregate(
        self, 
        claims: ClaimsResponse, 
        verification: VerificationList
    ) -> IntegratedReport:
        """Merge claims and verification into integrated report
        
        Args:
            claims: Extracted claims
            verification: Verification results
            
        Returns:
            IntegratedReport with complete analysis
        """
        # Create a mapping of claim_id to verification result
        verification_map = {
            v.claim_id: v for v in verification.verification_results
        }
        
        # Build integrated results
        integrated_results = []
        for claim in claims.claims:
            verification_result = verification_map.get(claim.id)
            if not verification_result:
                continue
                
            integrated_result = self._build_integrated_result(claim, verification_result)
            integrated_results.append(integrated_result)
        
        # Generate summary
        summary = self._generate_summary(integrated_results)
        
        # Build final report
        report = IntegratedReport(
            company_name=self.company_name,
            processed_at=datetime.now().isoformat(),
            summary=summary,
            results=integrated_results
        )
        
        return report
    
    def _build_integrated_result(self, claim, verification_result) -> IntegratedClaimResult:
        """Build single integrated claim result"""
        
        # Classify evidence type
        evidence_type = self._classify_evidence_type(
            verification_result.verdict, 
            len(verification_result.sources)
        )
        
        # Build enhanced verification
        enhanced_verification = EnhancedVerification(
            verdict=verification_result.verdict,
            confidence=verification_result.confidence,
            reasoning=verification_result.reasoning,
            evidence_type=evidence_type,
            source_count=len(verification_result.sources),
            sources=verification_result.sources,
            contradiction_details=self._get_contradiction_details(
                verification_result.verdict,
                verification_result.reasoning
            ) if verification_result.verdict == Verdict.CONTRADICTED else None
        )
        
        # Assess risk
        risk_assessment = self._assess_risk(
            claim.claim,
            verification_result.verdict,
            verification_result.confidence,
            evidence_type
        )
        
        # Categorize claim
        category = self._categorize_claim(claim.claim)
        
        return IntegratedClaimResult(
            claim_id=claim.id,
            claim_text=claim.claim,
            category=category,
            extraction_confidence=claim.confidence,
            original_evidence=claim.evidence,
            verification=enhanced_verification,
            risk_assessment=risk_assessment
        )
    
    def _classify_evidence_type(self, verdict: Verdict, source_count: int) -> EvidenceType:
        """Classify evidence type based on verdict and sources"""
        if source_count == 0:
            return EvidenceType.NO_EVIDENCE
        elif verdict == Verdict.CONTRADICTED:
            return EvidenceType.CONFLICTING_SOURCES
        elif verdict == Verdict.SUPPORTED:
            return EvidenceType.CONSISTENT_SOURCES
        else:  # INSUFFICIENT_EVIDENCE with some sources
            return EvidenceType.NO_EVIDENCE
    
    def _get_contradiction_details(self, verdict: Verdict, reasoning: str) -> str:
        """Extract contradiction details from reasoning"""
        if verdict != Verdict.CONTRADICTED:
            return None
        # For now, just return the reasoning
        # In future, could use LLM to extract specific contradictions
        return reasoning
    
    def _assess_risk(
        self, 
        claim_text: str, 
        verdict: Verdict, 
        confidence: float,
        evidence_type: EvidenceType
    ) -> RiskAssessment:
        """Assess investment risk for a claim"""
        
        # Determine risk level
        if verdict == Verdict.CONTRADICTED:
            risk_level = RiskLevel.HIGH
        elif verdict == Verdict.INSUFFICIENT_EVIDENCE or evidence_type == EvidenceType.NO_EVIDENCE:
            risk_level = RiskLevel.MEDIUM
        elif verdict == Verdict.SUPPORTED and confidence >= 0.8:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MEDIUM
        
        # Set investor alert
        investor_alert = (
            verdict == Verdict.CONTRADICTED or 
            (verdict == Verdict.INSUFFICIENT_EVIDENCE and self._is_critical_claim(claim_text))
        )
        
        # Determine if follow-up needed
        requires_followup = risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]
        
        # Generate follow-up questions
        followup_questions = self._generate_followup_questions(
            claim_text, verdict, evidence_type
        ) if requires_followup else []
        
        return RiskAssessment(
            risk_level=risk_level,
            investor_alert=investor_alert,
            requires_followup=requires_followup,
            followup_questions=followup_questions
        )
    
    def _is_critical_claim(self, claim_text: str) -> bool:
        """Check if claim is critical for investment decision"""
        critical_keywords = [
            'revenue', 'profit', 'funding', 'valuation', 'market share',
            'partnership', 'contract', 'patent', 'regulatory', 'compliance'
        ]
        claim_lower = claim_text.lower()
        return any(keyword in claim_lower for keyword in critical_keywords)
    
    def _generate_followup_questions(
        self, 
        claim_text: str, 
        verdict: Verdict,
        evidence_type: EvidenceType
    ) -> List[str]:
        """Generate follow-up questions for due diligence"""
        questions = []
        
        if verdict == Verdict.CONTRADICTED:
            questions.append("Request official documentation to clarify discrepancies")
            questions.append("Verify data sources and methodology used")
        elif evidence_type == EvidenceType.NO_EVIDENCE:
            questions.append("Request supporting evidence and documentation")
            questions.append("Verify claim with independent third-party sources")
        
        # Add specific questions based on claim type
        claim_lower = claim_text.lower()
        if 'revenue' in claim_lower or 'financial' in claim_lower:
            questions.append("Request audited financial statements")
        if 'partnership' in claim_lower:
            questions.append("Request partnership agreement or press release")
        if 'market' in claim_lower:
            questions.append("Request market research report or analysis")
        
        return questions[:3]  # Limit to top 3 questions
    
    def _categorize_claim(self, claim_text: str) -> ClaimCategory:
        """Auto-categorize claim based on content"""
        claim_lower = claim_text.lower()
        
        # Financial indicators
        if any(word in claim_lower for word in ['revenue', 'profit', 'funding', 'valuation', 'financial']):
            return ClaimCategory.FINANCIAL
        
        # Market position indicators
        if any(word in claim_lower for word in ['market share', 'market size', 'market leader', 'competitor']):
            return ClaimCategory.MARKET_POSITION
        
        # Technology indicators
        if any(word in claim_lower for word in ['technology', 'patent', 'algorithm', 'ai', 'ml', 'software']):
            return ClaimCategory.TECHNOLOGY
        
        # Partnership indicators
        if any(word in claim_lower for word in ['partnership', 'collaboration', 'agreement', 'contract']):
            return ClaimCategory.PARTNERSHIPS
        
        # Team indicators
        if any(word in claim_lower for word in ['team', 'founder', 'ceo', 'employee', 'hire']):
            return ClaimCategory.TEAM
        
        # Product indicators
        if any(word in claim_lower for word in ['product', 'feature', 'service', 'platform', 'solution']):
            return ClaimCategory.PRODUCT
        
        # Regulatory indicators
        if any(word in claim_lower for word in ['regulatory', 'compliance', 'license', 'approval', 'certification']):
            return ClaimCategory.REGULATORY
        
        return ClaimCategory.OTHER
    
    def _generate_summary(self, results: List[IntegratedClaimResult]) -> ResultSummary:
        """Generate summary statistics"""
        total_claims = len(results)
        supported = sum(1 for r in results if r.verification.verdict == Verdict.SUPPORTED)
        contradicted = sum(1 for r in results if r.verification.verdict == Verdict.CONTRADICTED)
        insufficient = sum(1 for r in results if r.verification.verdict == Verdict.INSUFFICIENT_EVIDENCE)
        high_risk = sum(1 for r in results if r.risk_assessment.risk_level == RiskLevel.HIGH)
        
        # Evidence breakdown
        evidence_breakdown = {
            "no_evidence": sum(1 for r in results if r.verification.evidence_type == EvidenceType.NO_EVIDENCE),
            "conflicting_sources": sum(1 for r in results if r.verification.evidence_type == EvidenceType.CONFLICTING_SOURCES),
            "consistent_sources": sum(1 for r in results if r.verification.evidence_type == EvidenceType.CONSISTENT_SOURCES)
        }
        
        return ResultSummary(
            total_claims=total_claims,
            supported=supported,
            contradicted=contradicted,
            insufficient_evidence=insufficient,
            high_risk_count=high_risk,
            evidence_breakdown=evidence_breakdown
        )
    
    def save_report(self, report: IntegratedReport, path: str):
        """Save integrated report to JSON file"""
        report_dict = report.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
