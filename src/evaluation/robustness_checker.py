"""
Option 3: Robustness Checker

Tests verification stability by running verification multiple times with 
different prompt variations and measuring consistency.

This is fully automated - no manual labeling required.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics

from evaluation.claims_evaluation_schemes import (
    RobustnessResult,
    RobustnessReport
)
from evaluation.prompt_variations import get_prompt_variation, get_all_variations


class RobustnessChecker:
    """
    Checks robustness of claim verification by running multiple iterations
    with prompt variations.
    
    A robust verification should:
    - Give the same verdict regardless of prompt wording
    - Have low variance in certainty scores
    
    Metrics:
    - Consistency rate: % of claims with same verdict across runs
    - Certainty variance: How much certainty fluctuates
    - Identification of unstable claims
    """
    
    # Available prompt variations
    PROMPT_VARIATIONS = ["neutral", "evidence_focused", "skeptical"]
    
    def __init__(self, api_key: str = None, model: str = "gpt-5.2", temperature: float = 0):
        """
        Initialize the robustness checker.
        
        Args:
            api_key: OpenAI API key for running verifications
            model: Model to use for verification
            temperature: Temperature setting (default 0 for deterministic)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.verifier = None
    
    def setup(self):
        """Setup the internal verifier for live testing."""
        if not self.api_key:
            raise ValueError("API key required for live robustness testing")
        
        from claim_verification.claim_verifier import ClaimVerifier
        self.verifier = ClaimVerifier(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature
        )
        self.verifier.setup()
    
    def run_robustness_test(
        self, 
        claims,
        variations: List[str] = None
    ) -> RobustnessReport:
        """
        Run verification with multiple prompt variations and analyze consistency.
        
        Args:
            claims: ClaimsResponse object with claims to verify
            variations: List of prompt variation names to test (default: all 3)
            
        Returns:
            RobustnessReport with stability metrics
        """
        if self.verifier is None:
            self.setup()
        
        if variations is None:
            variations = self.PROMPT_VARIATIONS
        
        print(f"\n🔄 Running robustness test with {len(variations)} prompt variations...")
        print(f"   Claims to verify: {len(claims.claims)}")
        
        verification_runs = []
        
        for i, variation in enumerate(variations, 1):
            import time
            start_time = time.time()
            
            print(f"\n   [{i}/{len(variations)}] {variation.upper()}...")
            
            # Get the prompt variation
            system_prompt = get_prompt_variation(variation)
            
            # Temporarily override the system prompt
            original_prompt_method = self.verifier._get_system_prompt
            self.verifier._get_system_prompt = lambda sp=system_prompt: sp
            
            try:
                # Re-setup with new prompt
                self.verifier.setup()
                
                # Run verification
                verification_result, _ = self.verifier.verify_claims(claims)
                
                # Show results for this run
                verdicts = [r.verdict.value if hasattr(r.verdict, 'value') else r.verdict 
                           for r in verification_result.verification_results]
                supported = verdicts.count("SUPPORTED")
                contradicted = verdicts.count("CONTRADICTED")
                insufficient = verdicts.count("INSUFFICIENT_EVIDENCE")
                
                elapsed = time.time() - start_time
                print(f"       ✅{supported} ❌{contradicted} ⚠️{insufficient} ({elapsed:.1f}s)")
                
                # Convert to dict format
                run_data = {
                    "variation": variation,
                    "verification_results": [
                        {
                            "claim_id": r.claim_id,
                            "verdict": r.verdict.value if hasattr(r.verdict, 'value') else r.verdict,
                            "certainty": r.certainty,
                            "reasoning": r.reasoning
                        }
                        for r in verification_result.verification_results
                    ]
                }
                verification_runs.append(run_data)
                
            finally:
                # Restore original prompt
                self.verifier._get_system_prompt = original_prompt_method
        
        print(f"\n   ✅ All {len(variations)} variations completed!")
        
        # Analyze the runs
        return self.analyze_from_multiple_runs(verification_runs)
    
    def analyze_from_multiple_runs(
        self,
        verification_runs: List[Dict[str, Any]]
    ) -> RobustnessReport:
        """
        Analyze robustness from pre-existing verification runs.
        
        Args:
            verification_runs: List of verification results from different runs.
                Each run is a dict with verification_results list.
        
        Returns:
            RobustnessReport with stability metrics
        """
        if not verification_runs:
            return self._empty_report()
        
        num_runs = len(verification_runs)
        
        # Collect verdicts/certainties per claim across runs
        claim_data: Dict[str, Dict[str, Any]] = {}
        
        for run_idx, run in enumerate(verification_runs):
            results = run.get("verification_results", run.get("results", []))
            
            for result in results:
                claim_id = result.get("claim_id", "")
                verdict = result.get("verdict", "UNKNOWN")
                certainty = result.get("certainty", 0.0)
                claim_text = result.get("claim_text", result.get("claim", ""))
                
                if claim_id not in claim_data:
                    claim_data[claim_id] = {
                        "claim_text": claim_text,
                        "verdicts": [],
                        "certainties": []
                    }
                
                claim_data[claim_id]["verdicts"].append(verdict)
                claim_data[claim_id]["certainties"].append(certainty)
        
        return self._build_report(claim_data, num_runs)
    
    def analyze_from_files(self, file_paths: List[str]) -> RobustnessReport:
        """
        Analyze robustness from multiple verification JSON files.
        
        Args:
            file_paths: List of paths to verification result files
        """
        runs = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                runs.append(json.load(f))
        
        return self.analyze_from_multiple_runs(runs)
    
    def _build_report(
        self, 
        claim_data: Dict[str, Dict[str, Any]], 
        num_runs: int
    ) -> RobustnessReport:
        """Build robustness report from claim data."""
        claim_results = []
        unstable_claims = []
        stability_counts: Dict[str, List[bool]] = {}
        all_variances = []
        
        for claim_id, data in claim_data.items():
            verdicts = data["verdicts"]
            certainties = data["certainties"]
            claim_text = data["claim_text"]
            
            # Calculate consistency
            verdict_counts = Counter(verdicts)
            majority_verdict = verdict_counts.most_common(1)[0][0]
            consistency_rate = verdict_counts[majority_verdict] / len(verdicts)
            is_stable = consistency_rate == 1.0  # All runs agreed
            
            # Calculate certainty variance
            if len(certainties) > 1:
                cert_variance = statistics.variance(certainties)
            else:
                cert_variance = 0.0
            
            all_variances.append(cert_variance)
            
            result = RobustnessResult(
                claim_id=claim_id,
                claim_text=claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
                verdicts_across_runs=verdicts,
                certainties_across_runs=certainties,
                is_stable=is_stable,
                consistency_rate=consistency_rate,
                majority_verdict=majority_verdict,
                certainty_variance=round(cert_variance, 4)
            )
            
            claim_results.append(result)
            
            if not is_stable:
                unstable_claims.append(result)
            
            # Track stability by verdict type
            if majority_verdict not in stability_counts:
                stability_counts[majority_verdict] = []
            stability_counts[majority_verdict].append(is_stable)
        
        # Calculate overall metrics
        total_claims = len(claim_results)
        stable_count = sum(1 for r in claim_results if r.is_stable)
        overall_stability = stable_count / total_claims if total_claims > 0 else 0.0
        
        avg_variance = sum(all_variances) / len(all_variances) if all_variances else 0.0
        
        # Stability by verdict
        stability_by_verdict = {}
        for verdict, stable_list in stability_counts.items():
            rate = sum(stable_list) / len(stable_list) if stable_list else 0.0
            stability_by_verdict[verdict] = round(rate, 3)
        
        return RobustnessReport(
            analyzed_at=datetime.now().isoformat(),
            num_runs=num_runs,
            total_claims=total_claims,
            overall_stability_rate=round(overall_stability, 3),
            avg_certainty_variance=round(avg_variance, 4),
            stability_by_verdict=stability_by_verdict,
            unstable_claims=unstable_claims,
            claim_results=claim_results
        )
    
    def _empty_report(self) -> RobustnessReport:
        """Return empty report."""
        return RobustnessReport(
            analyzed_at=datetime.now().isoformat(),
            num_runs=0,
            total_claims=0,
            overall_stability_rate=0.0,
            avg_certainty_variance=0.0,
            stability_by_verdict={},
            unstable_claims=[],
            claim_results=[]
        )
    
    def save_report(self, report: RobustnessReport, output_path: str):
        """Save report to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"💾 Robustness report saved to: {output_path}")
    
    def print_summary(self, report: RobustnessReport):
        """Print human-readable summary."""
        # Header
        print("\n")
        print("╔" + "═"*58 + "╗")
        print("║" + " 🔄 ROBUSTNESS ANALYSIS REPORT (Option 3) ".center(58) + "║")
        print("╚" + "═"*58 + "╝")
        
        # Overall metrics
        print("\n┌─ Overall Stability " + "─"*38 + "┐")
        print(f"│  Verification Runs: {report.num_runs}".ljust(58) + "│")
        print(f"│  Total Claims: {report.total_claims}".ljust(58) + "│")
        print("├" + "─"*58 + "┤")
        
        # Stability with rating
        stability = report.overall_stability_rate
        if stability >= 0.95:
            rating = "🟢 Excellent"
        elif stability >= 0.80:
            rating = "🟡 Good"
        elif stability >= 0.60:
            rating = "🟠 Moderate"
        else:
            rating = "🔴 Poor"
        
        stable_count = int(report.total_claims * stability)
        print(f"│  Stability Rate: {stability:>6.1%}  ({stable_count}/{report.total_claims} stable)".ljust(58) + "│")
        print(f"│  Rating: {rating}".ljust(58) + "│")
        print(f"│  Certainty Variance: {report.avg_certainty_variance:.4f}".ljust(58) + "│")
        print("└" + "─"*58 + "┘")
        
        # Stability by verdict
        if report.stability_by_verdict:
            print("\n┌─ Stability by Verdict " + "─"*35 + "┐")
            print(f"│  {'Verdict':<30} {'Stability':>15}".ljust(58) + "│")
            print("├" + "─"*58 + "┤")
            for verdict, rate in report.stability_by_verdict.items():
                icon = "🟢" if rate >= 0.9 else "🟡" if rate >= 0.7 else "🔴"
                print(f"│  {icon} {verdict:<28} {rate:>14.0%}".ljust(58) + "│")
            print("└" + "─"*58 + "┘")
        
        # Unstable claims
        if report.unstable_claims:
            print(f"\n⚠️  Unstable Claims ({len(report.unstable_claims)}):")
            for claim in report.unstable_claims[:5]:
                verdicts_str = " → ".join(claim.verdicts_across_runs[:3])
                print(f"   • {claim.claim_id}: {verdicts_str} ({claim.consistency_rate:.0%})")
        else:
            print("\n✅ All claims are stable across runs!")
        
        print()


def main():
    """CLI entry point."""
    import argparse
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Check robustness of claim verifications")
    
    # Mode 1: Analyze existing files
    parser.add_argument("--files", nargs="+", 
                        help="Paths to verification result files from different runs")
    
    # Mode 2: Run live test with claims
    parser.add_argument("--claims", help="Path to claims.json for live robustness testing")
    parser.add_argument("--model", default="gpt-5.2", help="Model to use for verification")
    
    parser.add_argument("--output", default="res/evaluation/robustness_report.json", help="Output path")
    args = parser.parse_args()
    
    if not args.files and not args.claims:
        parser.error("Either --files or --claims must be specified")
    
    if args.claims:
        # Mode 2: Live robustness test
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return
        
        # Load claims
        from common.schemes import ClaimsResponse
        with open(args.claims, 'r', encoding='utf-8') as f:
            claims_data = json.load(f)
        claims = ClaimsResponse.model_validate(claims_data)
        
        # Run robustness test
        checker = RobustnessChecker(api_key=api_key, model=args.model)
        report = checker.run_robustness_test(claims)
    else:
        # Mode 1: Analyze existing files
        checker = RobustnessChecker()
        report = checker.analyze_from_files(args.files)
    
    checker.print_summary(report)
    checker.save_report(report, args.output)


if __name__ == "__main__":
    main()

