import yaml
import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from claim_verification.claim_verifier import ClaimVerifier
from claim_extractor.claim_extractor import ClaimExtractor
from file_content_extraction.ingestion_pipeline import IngestionPipeline
from file_content_extraction.data_loader import DataLoader
from common.result_aggregator import ResultAggregator
from common.schemes import MultiCompanyReport, ResultSummary, ClaimsResponse
from quality_checker.quality_checker import QualityChecker, QualityReport

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv()

def load_configuration():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    return config


def ingestion_pipeline():
    config = load_configuration()
    
    # Resolve paths relative to project root
    ingestion_pipeline = IngestionPipeline(
        csv_path=str(PROJECT_ROOT / config['file_content_extractor']['csv_path']),
        pdf_folder=str(PROJECT_ROOT / config['file_content_extractor']['pdf_folder']),
        output_path=str(PROJECT_ROOT / config['file_content_extractor']['output_path']),
        limit=config['file_content_extractor']['limit']
    )

    ingestion_pipeline.run()

def data_loader():
    config = load_configuration()
    
    # Resolve paths relative to project root
    data_loader = DataLoader(
        json_path=str(PROJECT_ROOT / config['file_content_extractor']['output_path'])
    )
    
    return data_loader

def claim_extraction():
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_configuration()
    
    claim_extractor = ClaimExtractor(
        api_key=api_key,
        model=config['claim_extractor']['model'],
        temperature=config['claim_extractor']['temperature'],
        max_claims=config['claim_extractor']['max_claims']
    )
    # Perform claim extractor setup
    claim_extractor.setup()
    
    for idx, item in enumerate(data_loader(), 1):
        # Extract company name from metadata
        company_name = item.metadata.get("Account Name", f"Company {idx}")

        print(f"\n--- Processing company {company_name}, ID {idx} ---")
        
        print(f"📝 Extracting claims...")
        claims = claim_extractor.extract_claims(item)
        print(f"   Found {len(claims.claims)} claims")
        
        print(f"💾 Saving extracted claims...")
        claim_extractor.store_as_json(
            claims,
            path=str(PROJECT_ROOT / "res" / f"claims_{idx}.json")
        )
    
    # Save all company claims extraction reports

def claim_verification():
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_configuration()
    
    
    claim_verifier = ClaimVerifier(
        api_key=api_key,
        model=config['claim_verifier']['model'],
        temperature=config['claim_verifier']['temperature']
    )
    # Perform claim verifier setup 
    claim_verifier.setup()

    # check the extracted claims
    results = Path(f"{PROJECT_ROOT}/res")

    if not results.exists():
        raise FileNotFoundError(f"Extraction results not found in {results}")
    
    if not results.is_dir():
        raise NotADirectoryError(f"Extraction results {results} is not a directory")

    claims_files = list(results.glob("claims_*.json"))

    if not claims_files:
        raise FileNotFoundError(f"No claims files found in {results}")
    
    # check if there are any suggested improvements in the quality report of the last round
    quality_output_path = Path(PROJECT_ROOT / config['quality_checker']['output_path'])
    quality_report_files = list(quality_output_path.glob("quality_report.json"))
    quality_report = None

    if quality_report_files:
        # may be we need to sort different quality reports to get the latest
        last_quality_report = quality_report_files[-1]
        with open(last_quality_report, "r") as f:
            quality_report = json.load(f)
            quality_report = QualityReport.model_validate(quality_report)
    
    all_search_results = []
    all_company_reports = []

    for claims_file in claims_files:
        with open(claims_file, "r") as f:
            claims = json.load(f)
            claims = ClaimsResponse.model_validate(claims)

        idx = claims_file.stem.split("_")[-1]
        company_name = f"Company {idx}"
        
        # Run claim verification
        verification_response, search_results = claim_verifier.verify_claims_with_improvements(claims, quality_report)
        
        # Collect search results for quality analysis
        all_search_results.append({
            "company": company_name,
            "search_results": search_results.model_dump() if search_results else None
        })
        
        print(f"💾 Saving verification results...")
        claim_verifier.store_as_json(
            verification_response, 
            path=Path(f"{PROJECT_ROOT}/res/verification_{idx}.json")
        )
        
        # Generate integrated report for this company
        print(f"📊 Generating integrated report...")
        aggregator = ResultAggregator(company_name=company_name)
        integrated_report = aggregator.aggregate(claims, verification_response)

        all_company_reports.append(integrated_report)
        
        # Print summary
        print(f"\n📈 Summary for {company_name}:")
        print(f"   Total Claims: {integrated_report.summary.total_claims}")
        print(f"   ✅ Supported: {integrated_report.summary.supported}")
        print(f"   ❌ Contradicted: {integrated_report.summary.contradicted}")
        print(f"   ⚠️  Insufficient Evidence: {integrated_report.summary.insufficient_evidence}")
        print(f"   🚨 High Risk: {integrated_report.summary.high_risk_count}")
        print(f"   Evidence: {integrated_report.summary.evidence_breakdown}")
        
        print(f"✅ Company {idx} completed!")

    return all_search_results, all_company_reports

def summary_verifications(all_company_reports):
    total_claims = sum(r.summary.total_claims for r in all_company_reports)
    total_supported = sum(r.summary.supported for r in all_company_reports)
    total_contradicted = sum(r.summary.contradicted for r in all_company_reports)
    total_insufficient = sum(r.summary.insufficient_evidence for r in all_company_reports)
    total_high_risk = sum(r.summary.high_risk_count for r in all_company_reports)
    
    # Aggregate evidence breakdown
    overall_evidence = {
        "no_evidence": sum(r.summary.evidence_breakdown.get("no_evidence", 0) for r in all_company_reports),
        "conflicting_sources": sum(r.summary.evidence_breakdown.get("conflicting_sources", 0) for r in all_company_reports),
        "consistent_sources": sum(r.summary.evidence_breakdown.get("consistent_sources", 0) for r in all_company_reports)
    }
    
    overall_summary = ResultSummary(
        total_claims=total_claims,
        supported=total_supported,
        contradicted=total_contradicted,
        insufficient_evidence=total_insufficient,
        high_risk_count=total_high_risk,
        evidence_breakdown=overall_evidence
    )
    
    # Create multi-company report
    multi_report = MultiCompanyReport(
        processed_at=datetime.now().isoformat(),
        total_companies=len(all_company_reports),
        overall_summary=overall_summary,
        companies=all_company_reports
    )
    
    # Save multi-company report
    print(f"💾 Saving multi-company final report...")
    final_report_path = str(PROJECT_ROOT / "res" / "final_report.json")
    os.makedirs(os.path.dirname(final_report_path), exist_ok=True)
    
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(multi_report.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Print overall summary
    print("\n" + "="*60)
    print("🎉 All processing completed!")
    print("="*60)
    print(f"\n📊 Overall Summary Across {len(all_company_reports)} Companies:")
    print(f"   Total Claims: {total_claims}")
    print(f"   ✅ Supported: {total_supported}")
    print(f"   ❌ Contradicted: {total_contradicted}")
    print(f"   ⚠️  Insufficient Evidence: {total_insufficient}")
    print(f"   🚨 High Risk: {total_high_risk}")
    print(f"   Evidence Breakdown: {overall_evidence}")
    print(f"\n📄 Final multi-company report saved to: res/final_report.json")

    return final_report_path

def quality_assessment(all_search_results, final_report_path):
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_configuration()
    
    # Initialize quality checker
    quality_checker = QualityChecker(
        api_key=api_key,
        model=config['quality_checker']['model'],
        temperature=config['quality_checker']['temperature']
    )
    quality_checker.setup()
    
    print("📊 Analyzing verification quality...")
    quality_report, search_quality_summary = quality_checker.check_quality(
        report_path=final_report_path,
        search_results=all_search_results
    )
    
    # Save quality report with search quality summary
    quality_output_path = str(PROJECT_ROOT / config['quality_checker']['output_path'])
    quality_checker.store_as_json(quality_report, quality_output_path, search_quality_summary)
    # Print quality summary
    print("\n📈 Quality Assessment Results:")
    print(f"   Overall Score: {quality_report.overall_quality_score:.2f}")
    print(f"   Overall Rating: {quality_report.overall_rating.value.upper()}")
    print(f"   Critical Issues: {quality_report.critical_issues_count}")
    
    # Print search quality if available
    if search_quality_summary:
        print(f"\n🔍 Search Quality:")
        print(f"   Total Searches: {search_quality_summary.get('total_searches', 0)}")
        print(f"   Good Rate: {search_quality_summary.get('good_rate', 0):.0%}")
        print(f"   Failed Rate: {search_quality_summary.get('failed_rate', 0):.0%}")
        print(f"   Off-Topic Rate: {search_quality_summary.get('off_topic_rate', 0):.0%}")
    
    if quality_report.top_issues:
        print("\n🔴 Top Issues to Address:")
        for i, issue in enumerate(quality_report.top_issues[:3], 1):
            print(f"   {i}. {issue}")
    
    print(f"\n📄 Quality report saved to: {config['quality_checker']['output_path']}")

    return quality_output_path


def main_pipeline():
    MAX_ROUNDS = 3
    ingestion_pipeline()
    claim_extraction()
    for i in range(MAX_ROUNDS):
        print(f"🚀 Starting Round {i+1}...")
        all_search_results, all_company_reports = claim_verification()
        final_report_path = summary_verifications(all_company_reports)
        quality_output_path = quality_assessment(all_search_results, final_report_path)
        print(f"🎉 Round {i+1} completed!")


if __name__ == "__main__":
    main_pipeline()