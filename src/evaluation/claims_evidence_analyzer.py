"""
Option 2: Evidence Justification Analyzer

Analyzes the quality and quantity of evidence used to justify each claim verification.
This is fully automated - no manual labeling required.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from collections import defaultdict

from evaluation.claims_evaluation_schemes import (
    ClaimEvidenceStats,
    EvidenceAnalysisReport
)


class ClaimsEvidenceAnalyzer:
    """
    Analyzes evidence quality for claim verifications.
    
    Metrics calculated:
    - Number of sources per claim
    - Distribution of source types
    - Correlation between evidence quantity and verdict
    - Flags claims with low evidence for their certainty level
    """
    
    # Minimum sources expected for high-certainty claims
    HIGH_CERTAINTY_THRESHOLD = 0.7
    MIN_SOURCES_FOR_HIGH_CERTAINTY = 2
    
    def __init__(self):
        self.source_type_patterns = {
            "news": ["reuters", "bloomberg", "techcrunch", "forbes", "wsj", "nytimes"],
            "official": ["linkedin", "crunchbase", "pitchbook"],
            "regulatory": ["sec.gov", "bafin", "fca.org"],
            "social": ["twitter", "reddit", "linkedin"],
            "company": ["company website", "press release"]
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return url
    
    def _classify_source_type(self, url: str) -> str:
        """Classify a source URL by type."""
        url_lower = url.lower()
        domain = self._extract_domain(url_lower)
        
        for source_type, patterns in self.source_type_patterns.items():
            if any(pattern in domain or pattern in url_lower for pattern in patterns):
                return source_type
        
        return "other"
    
    def analyze_claim(
        self, 
        claim_id: str,
        claim_text: str,
        verdict: str,
        certainty: float,
        sources: List[str]
    ) -> ClaimEvidenceStats:
        """Analyze evidence for a single claim."""
        source_count = len(sources) if sources else 0
        
        # Count unique domains
        domains = set(self._extract_domain(url) for url in sources) if sources else set()
        unique_domains = len(domains)
        
        # Classify sources by type
        source_types: Dict[str, int] = defaultdict(int)
        for url in (sources or []):
            source_type = self._classify_source_type(url)
            source_types[source_type] += 1
        
        # Check if evidence is sufficient for the certainty level
        has_sufficient = True
        if certainty >= self.HIGH_CERTAINTY_THRESHOLD:
            has_sufficient = source_count >= self.MIN_SOURCES_FOR_HIGH_CERTAINTY
        
        return ClaimEvidenceStats(
            claim_id=claim_id,
            claim_text=claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
            verdict=verdict,
            certainty=certainty,
            source_count=source_count,
            unique_domains=unique_domains,
            source_types=dict(source_types),
            has_sufficient_evidence=has_sufficient
        )
    
    def analyze_from_report(self, final_report_path: str) -> EvidenceAnalysisReport:
        """
        Analyze evidence from a final_report.json file.
        
        Args:
            final_report_path: Path to the final report JSON
            
        Returns:
            EvidenceAnalysisReport with evidence quality metrics
        """
        with open(final_report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        claim_details = []
        verdict_source_counts: Dict[str, List[int]] = defaultdict(list)
        
        # Process each company's results
        companies = report.get("companies", [report])  # Handle single or multi-company
        
        for company in companies:
            results = company.get("results", [])
            for result in results:
                claim_id = result.get("claim_id", "")
                claim_text = result.get("claim_text", "")
                verification = result.get("verification", {})
                
                verdict = verification.get("verdict", "UNKNOWN")
                certainty = verification.get("certainty", 0.0)
                sources = verification.get("sources", [])
                
                stats = self.analyze_claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    verdict=verdict,
                    certainty=certainty,
                    sources=sources
                )
                claim_details.append(stats)
                verdict_source_counts[verdict].append(stats.source_count)
        
        return self._build_report(claim_details, verdict_source_counts)
    
    def analyze_from_verification(
        self, 
        claims_data: List[Dict[str, Any]],
        verification_data: List[Dict[str, Any]]
    ) -> EvidenceAnalysisReport:
        """
        Analyze evidence from claims and verification lists.
        
        Args:
            claims_data: List of claim dicts with id, claim
            verification_data: List of verification dicts with claim_id, verdict, certainty, sources
        """
        # Build claim lookup
        claim_lookup = {c.get("id", c.get("claim_id")): c.get("claim", "") for c in claims_data}
        
        claim_details = []
        verdict_source_counts: Dict[str, List[int]] = defaultdict(list)
        
        for v in verification_data:
            claim_id = v.get("claim_id", "")
            claim_text = claim_lookup.get(claim_id, "")
            verdict = v.get("verdict", "UNKNOWN")
            certainty = v.get("certainty", 0.0)
            sources = v.get("sources", [])
            
            stats = self.analyze_claim(
                claim_id=claim_id,
                claim_text=claim_text,
                verdict=verdict,
                certainty=certainty,
                sources=sources
            )
            claim_details.append(stats)
            verdict_source_counts[verdict].append(stats.source_count)
        
        return self._build_report(claim_details, verdict_source_counts)
    
    def _build_report(
        self, 
        claim_details: List[ClaimEvidenceStats],
        verdict_source_counts: Dict[str, List[int]]
    ) -> EvidenceAnalysisReport:
        """Build the final report from analyzed claims."""
        total_claims = len(claim_details)
        
        if total_claims == 0:
            return EvidenceAnalysisReport(
                analyzed_at=datetime.now().isoformat(),
                total_claims=0,
                avg_sources_per_claim=0.0,
                avg_sources_supported=0.0,
                avg_sources_contradicted=0.0,
                avg_sources_insufficient=0.0,
                claims_with_no_sources=0,
                claims_with_single_source=0,
                claims_with_multiple_sources=0,
                evidence_verdict_correlation={},
                claim_details=[],
                low_evidence_claims=[]
            )
        
        # Calculate averages
        all_sources = [c.source_count for c in claim_details]
        avg_sources = sum(all_sources) / len(all_sources)
        
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        avg_supported = safe_avg(verdict_source_counts.get("SUPPORTED", []))
        avg_contradicted = safe_avg(verdict_source_counts.get("CONTRADICTED", []))
        avg_insufficient = safe_avg(verdict_source_counts.get("INSUFFICIENT_EVIDENCE", []))
        
        # Count by source levels
        no_sources = sum(1 for c in claim_details if c.source_count == 0)
        single_source = sum(1 for c in claim_details if c.source_count == 1)
        multiple_sources = sum(1 for c in claim_details if c.source_count >= 2)
        
        # Evidence-verdict correlation
        correlation: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in claim_details:
            bucket = "0" if c.source_count == 0 else "1" if c.source_count == 1 else "2+" 
            correlation[bucket][c.verdict] += 1
        
        # Identify low-evidence claims
        low_evidence = [
            f"{c.claim_id}: {c.verdict} with certainty {c.certainty:.2f} but only {c.source_count} sources"
            for c in claim_details 
            if not c.has_sufficient_evidence
        ]
        
        return EvidenceAnalysisReport(
            analyzed_at=datetime.now().isoformat(),
            total_claims=total_claims,
            avg_sources_per_claim=round(avg_sources, 2),
            avg_sources_supported=round(avg_supported, 2),
            avg_sources_contradicted=round(avg_contradicted, 2),
            avg_sources_insufficient=round(avg_insufficient, 2),
            claims_with_no_sources=no_sources,
            claims_with_single_source=single_source,
            claims_with_multiple_sources=multiple_sources,
            evidence_verdict_correlation={k: dict(v) for k, v in correlation.items()},
            claim_details=claim_details,
            low_evidence_claims=low_evidence
        )
    
    def save_report(self, report: EvidenceAnalysisReport, output_path: str):
        """Save report to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"💾 Evidence analysis saved to: {output_path}")
    
    def print_summary(self, report: EvidenceAnalysisReport):
        """Print human-readable summary."""
        # Header
        print("\n")
        print("╔" + "═"*58 + "╗")
        print("║" + " 📊 EVIDENCE ANALYSIS REPORT (Option 2) ".center(58) + "║")
        print("╚" + "═"*58 + "╝")
        
        # Statistics box
        print("\n┌─ Overview " + "─"*47 + "┐")
        print(f"│  Total Claims Analyzed: {report.total_claims}".ljust(58) + "│")
        print(f"│  Average Sources/Claim: {report.avg_sources_per_claim:.2f}".ljust(58) + "│")
        print("└" + "─"*58 + "┘")
        
        # Source distribution with visual bar
        print("\n┌─ Source Distribution " + "─"*36 + "┐")
        total = report.total_claims or 1
        
        def bar(count, total, width=20):
            filled = int((count / total) * width) if total > 0 else 0
            return "█" * filled + "░" * (width - filled)
        
        no_pct = report.claims_with_no_sources / total
        single_pct = report.claims_with_single_source / total
        multi_pct = report.claims_with_multiple_sources / total
        
        print(f"│  No sources:   {bar(report.claims_with_no_sources, total)} {report.claims_with_no_sources:>3} ({no_pct:>5.0%})".ljust(58) + "│")
        print(f"│  Single (1):   {bar(report.claims_with_single_source, total)} {report.claims_with_single_source:>3} ({single_pct:>5.0%})".ljust(58) + "│")
        print(f"│  Multiple (2+):{bar(report.claims_with_multiple_sources, total)} {report.claims_with_multiple_sources:>3} ({multi_pct:>5.0%})".ljust(58) + "│")
        print("└" + "─"*58 + "┘")
        
        # Avg by verdict
        print("\n┌─ Avg Sources by Verdict " + "─"*33 + "┐")
        print(f"│  {'Verdict':<25} {'Avg Sources':>15}".ljust(58) + "│")
        print("├" + "─"*58 + "┤")
        print(f"│  {'✅ SUPPORTED':<25} {report.avg_sources_supported:>15.2f}".ljust(58) + "│")
        print(f"│  {'❌ CONTRADICTED':<25} {report.avg_sources_contradicted:>15.2f}".ljust(58) + "│")
        print(f"│  {'⚠️  INSUFFICIENT_EVIDENCE':<25} {report.avg_sources_insufficient:>15.2f}".ljust(58) + "│")
        print("└" + "─"*58 + "┘")
        
        # Warnings
        if report.low_evidence_claims:
            print(f"\n⚠️  Low Evidence Warnings ({len(report.low_evidence_claims)}):")
            for warning in report.low_evidence_claims[:3]:
                print(f"   • {warning[:55]}...")
        
        print()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze evidence quality for claim verifications")
    parser.add_argument("--report", required=True, help="Path to final_report.json")
    parser.add_argument("--output", default="res/evaluation/claims_evidence_report.json", help="Output path")
    args = parser.parse_args()
    
    analyzer = ClaimsEvidenceAnalyzer()
    report = analyzer.analyze_from_report(args.report)
    analyzer.print_summary(report)
    analyzer.save_report(report, args.output)


if __name__ == "__main__":
    main()
