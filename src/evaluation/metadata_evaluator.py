import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Any
from difflib import SequenceMatcher

from evaluation.evaluation_schemes import (
    FieldMetrics,
    CompanyEvaluationResult,
    MetadataEvaluationReport
)


class MetadataEvaluator:
    """
    Evaluates metadata extraction accuracy using precision/recall/F1 metrics.
    
    Compares extracted metadata against verified ground truth from Salesforce CSV.
    """
    
    def __init__(self, ground_truth_csv: str, identifier_field: str = "Account Name"):
        """
        Initialize the evaluator with ground truth data.
        
        Args:
            ground_truth_csv: Path to the Salesforce CSV with verified labels
            identifier_field: Field used to match companies (default: "Account Name")
        """
        self.ground_truth_csv = ground_truth_csv
        self.identifier_field = identifier_field
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self.fields: List[str] = []
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Load ground truth data from CSV into memory."""
        with open(self.ground_truth_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fields = reader.fieldnames or []
            
            for row in reader:
                identifier = row.get(self.identifier_field, "").strip()
                if identifier:
                    # Normalize identifier for matching
                    normalized_key = self._normalize_string(identifier)
                    self.ground_truth[normalized_key] = row
        
        print(f"📊 Loaded {len(self.ground_truth)} ground truth records")
        print(f"   Fields available: {len(self.fields)}")
    
    def _normalize_string(self, s: str) -> str:
        """Normalize string for comparison (lowercase, strip whitespace)."""
        if s is None:
            return ""
        return str(s).lower().strip()
    
    def _fuzzy_match(self, s1: str, s2: str, threshold: float = 0.85) -> bool:
        """Check if two strings are similar enough using fuzzy matching."""
        s1_norm = self._normalize_string(s1)
        s2_norm = self._normalize_string(s2)
        
        # Exact match
        if s1_norm == s2_norm:
            return True
        
        # Empty check
        if not s1_norm or not s2_norm:
            return s1_norm == s2_norm
        
        # Fuzzy match using SequenceMatcher
        ratio = SequenceMatcher(None, s1_norm, s2_norm).ratio()
        return ratio >= threshold
    
    def _find_ground_truth(self, company_identifier: str) -> Optional[Dict[str, Any]]:
        """Find ground truth record for a company using fuzzy matching."""
        normalized = self._normalize_string(company_identifier)
        
        # Try exact match first
        if normalized in self.ground_truth:
            return self.ground_truth[normalized]
        
        # Try fuzzy matching
        for key, record in self.ground_truth.items():
            if self._fuzzy_match(company_identifier, key, threshold=0.8):
                return record
        
        return None
    
    def evaluate_company(
        self, 
        extracted_metadata: Dict[str, Any], 
        company_identifier: str
    ) -> CompanyEvaluationResult:
        """
        Evaluate extracted metadata for a single company.
        
        Args:
            extracted_metadata: Dict of extracted field values
            company_identifier: Company name/ID for lookup
            
        Returns:
            CompanyEvaluationResult with per-field comparison
        """
        ground_truth = self._find_ground_truth(company_identifier)
        
        if ground_truth is None:
            return CompanyEvaluationResult(
                company_identifier=company_identifier,
                matched=False,
                field_results={},
                accuracy=0.0
            )
        
        field_results = {}
        matches = 0
        total = 0
        
        for field in self.fields:
            extracted_value = extracted_metadata.get(field, "")
            expected_value = ground_truth.get(field, "")
            
            # Skip empty ground truth values
            if not expected_value or expected_value.strip() == "":
                continue
            
            total += 1
            is_match = self._fuzzy_match(str(extracted_value), str(expected_value))
            
            if is_match:
                matches += 1
            
            field_results[field] = {
                "extracted": str(extracted_value) if extracted_value else None,
                "expected": str(expected_value),
                "match": is_match
            }
        
        accuracy = matches / total if total > 0 else 0.0
        
        return CompanyEvaluationResult(
            company_identifier=company_identifier,
            matched=True,
            field_results=field_results,
            accuracy=accuracy
        )
    
    def evaluate_batch(
        self, 
        extracted_results: List[Dict[str, Any]],
        identifier_key: str = "company_name",
        categorical_fields: List[str] = None
    ) -> MetadataEvaluationReport:
        """
        Evaluate multiple companies and compute aggregate metrics.
        
        Args:
            extracted_results: List of dicts with extracted metadata
            identifier_key: Key in each dict containing company identifier
            categorical_fields: Fields to build confusion matrices for
            
        Returns:
            MetadataEvaluationReport with overall and per-field metrics
        """
        # Default categorical fields for confusion matrices
        if categorical_fields is None:
            categorical_fields = [
                "Startup Country",
                "Startup Industry Sector", 
                "UVC Investment Status",
                "Initial Impression"
            ]
        
        company_results = []
        field_stats: Dict[str, Dict[str, int]] = {}  # field -> {tp, fp, fn}
        
        # Confusion matrices: field -> {actual_label -> {predicted_label -> count}}
        confusion_data: Dict[str, Dict[str, Dict[str, int]]] = {
            field: {} for field in categorical_fields
        }
        
        for item in extracted_results:
            # Try multiple sources for company identifier
            metadata = item.get("metadata", {})
            
            # Priority: 1) metadata.Account Name, 2) identifier_key, 3) top-level Account Name
            company_id = (
                metadata.get("Account Name") or 
                item.get(identifier_key) or 
                item.get("Account Name") or 
                "Unknown"
            )
            
            # Use metadata for evaluation if available, otherwise use item directly
            eval_data = metadata if metadata else item
            
            result = self.evaluate_company(eval_data, company_id)
            company_results.append(result)
            
            # Aggregate field-level stats
            for field, comparison in result.field_results.items():
                if field not in field_stats:
                    field_stats[field] = {"tp": 0, "fp": 0, "fn": 0, "total": 0}
                
                field_stats[field]["total"] += 1
                
                if comparison["match"]:
                    field_stats[field]["tp"] += 1
                else:
                    # If we extracted something but it's wrong -> FP
                    # If ground truth exists but we miss it -> FN
                    if comparison["extracted"]:
                        field_stats[field]["fp"] += 1
                    else:
                        field_stats[field]["fn"] += 1
                
                # Build confusion matrix for categorical fields
                if field in categorical_fields:
                    actual = self._normalize_string(comparison["expected"]) or "(empty)"
                    predicted = self._normalize_string(comparison["extracted"] or "") or "(empty)"
                    
                    if actual not in confusion_data[field]:
                        confusion_data[field][actual] = {}
                    if predicted not in confusion_data[field][actual]:
                        confusion_data[field][actual][predicted] = 0
                    confusion_data[field][actual][predicted] += 1
        
        # Calculate per-field metrics
        field_metrics = []
        for field, stats in field_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            total = stats["total"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / total if total > 0 else 0.0
            
            field_metrics.append(FieldMetrics(
                field_name=field,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                total_samples=total
            ))
        
        # Sort by F1 score descending
        field_metrics.sort(key=lambda x: x.f1_score, reverse=True)
        
        # Calculate overall metrics
        total_tp = sum(s["tp"] for s in field_stats.values())
        total_fp = sum(s["fp"] for s in field_stats.values())
        total_fn = sum(s["fn"] for s in field_stats.values())
        total_samples = sum(s["total"] for s in field_stats.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_accuracy = total_tp / total_samples if total_samples > 0 else 0.0
        
        matched_count = sum(1 for r in company_results if r.matched)
        
        # Filter out empty confusion matrices
        confusion_matrices = {
            field: matrix for field, matrix in confusion_data.items() 
            if matrix  # Only include if there's data
        }
        
        return MetadataEvaluationReport(
            evaluated_at=datetime.now().isoformat(),
            total_companies=len(company_results),
            matched_companies=matched_count,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            overall_accuracy=overall_accuracy,
            field_metrics=field_metrics,
            company_results=company_results,
            confusion_matrices=confusion_matrices
        )
    
    def save_report(self, report: MetadataEvaluationReport, output_path: str):
        """Save evaluation report to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"💾 Evaluation report saved to: {output_path}")
    
    def print_summary(self, report: MetadataEvaluationReport):
        """Print a human-readable summary of the evaluation."""
        # Header
        print("\n")
        print("╔" + "═"*58 + "╗")
        print("║" + " 📊 METADATA EVALUATION REPORT ".center(58) + "║")
        print("╚" + "═"*58 + "╝")
        
        # Overall metrics box
        print("\n┌─ Overall Metrics " + "─"*40 + "┐")
        print(f"│  Companies Evaluated: {report.total_companies:<5}  Matched: {report.matched_companies:<5}".ljust(58) + "│")
        print("├" + "─"*58 + "┤")
        print(f"│  {'Metric':<15} {'Score':>10}  {'Rating':>15}".ljust(58) + "│")
        print("├" + "─"*58 + "┤")
        
        def rating(score):
            if score >= 0.9: return "🟢 Excellent"
            if score >= 0.7: return "🟡 Good"
            if score >= 0.5: return "🟠 Fair"
            return "🔴 Poor"
        
        print(f"│  {'Precision':<15} {report.overall_precision:>9.1%}   {rating(report.overall_precision):>15}".ljust(58) + "│")
        print(f"│  {'Recall':<15} {report.overall_recall:>9.1%}   {rating(report.overall_recall):>15}".ljust(58) + "│")
        print(f"│  {'F1 Score':<15} {report.overall_f1:>9.1%}   {rating(report.overall_f1):>15}".ljust(58) + "│")
        print(f"│  {'Accuracy':<15} {report.overall_accuracy:>9.1%}   {rating(report.overall_accuracy):>15}".ljust(58) + "│")
        print("└" + "─"*58 + "┘")
        
        # Field metrics table
        if report.field_metrics:
            print("\n┌─ Top 10 Fields by F1 Score " + "─"*30 + "┐")
            print(f"│  {'#':<3} {'Field':<25} {'P':>7} {'R':>7} {'F1':>7}".ljust(58) + "│")
            print("├" + "─"*58 + "┤")
            
            for i, fm in enumerate(report.field_metrics[:10], 1):
                name = fm.field_name[:24] if len(fm.field_name) > 24 else fm.field_name
                line = f"│  {i:<3} {name:<25} {fm.precision:>6.0%} {fm.recall:>6.0%} {fm.f1_score:>6.0%}"
                print(line.ljust(58) + "│")
            print("└" + "─"*58 + "┘")
            
            # Bottom fields
            if len(report.field_metrics) > 10:
                print("\n⚠️  Lowest Performing Fields:")
                for fm in report.field_metrics[-3:]:
                    name = fm.field_name[:30] if len(fm.field_name) > 30 else fm.field_name
                    print(f"   • {name:<30} F1={fm.f1_score:.0%}")
        
        # Confusion matrices (compact)
        if report.confusion_matrices:
            print("\n┌─ Confusion Matrices " + "─"*37 + "┐")
            for field, matrix in report.confusion_matrices.items():
                correct = sum(matrix.get(k, {}).get(k, 0) for k in matrix)
                total = sum(sum(v.values()) for v in matrix.values())
                acc = correct / total if total > 0 else 0
                print(f"│  {field[:35]:<35} {correct}/{total} ({acc:.0%})".ljust(58) + "│")
            print("└" + "─"*58 + "┘")
        
        print()


def main():
    """CLI entry point for running metadata evaluation."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Evaluate metadata extraction accuracy")
    parser.add_argument("--csv", required=True, help="Path to ground truth CSV")
    parser.add_argument("--results", required=True, help="Path to extracted results JSON")
    parser.add_argument("--output", default="res/evaluation/metadata_output.json", help="Output path for report")
    args = parser.parse_args()
    
    # Load extracted results
    with open(args.results, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    # Handle different data formats
    if isinstance(extracted_data, dict):
        # If it's a single dict, wrap it
        if "companies" in extracted_data:
            results_list = extracted_data["companies"]
        else:
            results_list = [extracted_data]
    else:
        results_list = extracted_data
    
    # Run evaluation
    evaluator = MetadataEvaluator(ground_truth_csv=args.csv)
    report = evaluator.evaluate_batch(results_list)
    
    # Print and save
    evaluator.print_summary(report)
    evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main()
