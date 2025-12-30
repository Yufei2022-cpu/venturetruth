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
from common.schemes import MultiCompanyReport, ResultSummary

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv()

def load_configuration():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    return config
    

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_configuration()
    
    # Resolve paths relative to project root
    ingestion_pipeline = IngestionPipeline(
        csv_path=str(PROJECT_ROOT / config['file_content_extractor']['csv_path']),
        pdf_folder=str(PROJECT_ROOT / config['file_content_extractor']['pdf_folder']),
        output_path=str(PROJECT_ROOT / config['file_content_extractor']['output_path']),
        limit=config['file_content_extractor']['limit']
    )
    
    data_loader = DataLoader(
        json_path=str(PROJECT_ROOT / config['file_content_extractor']['output_path'])
    )
    
    claim_extractor = ClaimExtractor(
        api_key=api_key,
        model=config['claim_extractor']['model'],
        temperature=config['claim_extractor']['temperature'],
        max_claims=config['claim_extractor']['max_claims']
    )
    # Perform claim extractor setup
    claim_extractor.setup()
    
    claim_verifier = ClaimVerifier(
        api_key=api_key,
        model=config['claim_verifier']['model'],
        temperature=config['claim_verifier']['temperature']
    )
    # Perform claim verifier setup 
    claim_verifier.setup()
    
    # Process files and store them
    print("\n" + "="*60)
    print("📁 Stage 1: File Content Extraction")
    print("="*60)
    ingestion_pipeline.run()
    
    # Load the Information & perform claim extraction and verification
    print("\n" + "="*60)
    print("📋 Stage 2: Claim Extraction & Verification")
    print("="*60)
    
    # Collect all company reports
    all_company_reports = []
    
    for idx, item in enumerate(data_loader, 1):
        print(f"\n--- Processing company {idx} ---")
        
        # Extract company name from metadata
        company_name = item.metadata.get("Account Name", f"Company {idx}")
        
        print(f"📝 Extracting claims...")
        claims = claim_extractor.extract_claims(item)
        print(f"   Found {len(claims.claims)} claims")
        
        print(f"💾 Saving extracted claims...")
        claim_extractor.store_as_json(
            claims,
            path=str(PROJECT_ROOT / "res" / f"claims_{idx}.json")
        )
        
        print(f"🔎 Verifying claims...")
        verification_response = claim_verifier.verify_claims(claims)
        
        print(f"💾 Saving verification results...")
        claim_verifier.store_as_json(
            verification_response, 
            path=str(PROJECT_ROOT / "res" / f"verification_{idx}.json")
        )
        
        # Generate integrated report for this company
        print(f"📊 Generating integrated report...")
        aggregator = ResultAggregator(company_name=company_name)
        integrated_report = aggregator.aggregate(claims, verification_response)
        
        # Add to collection
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
    
    # Generate overall summary across all companies
    print("\n" + "="*60)
    print("📊 Generating Multi-Company Report")
    print("="*60)
    
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
    

if __name__ == "__main__":
    main()