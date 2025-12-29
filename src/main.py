import yaml
import os
from pathlib import Path

from dotenv import load_dotenv

from claim_verification.claim_verifier import ClaimVerifier
from claim_extractor.claim_extractor import ClaimExtractor
from file_content_extraction.ingestion_pipeline import IngestionPipeline
from file_content_extraction.data_loader import DataLoader
from common.result_aggregator import ResultAggregator

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
            path=str(PROJECT_ROOT / "res" / "claims.json")
        )
        
        print(f"🔎 Verifying claims...")
        verification_response = claim_verifier.verify_claims(claims)
        
        print(f"💾 Saving verification results...")
        claim_verifier.store_as_json(
            verification_response, 
            path=str(PROJECT_ROOT / "res" / "verification_response.json")
        )
        
        # Generate integrated report
        print(f"📊 Generating integrated report...")
        aggregator = ResultAggregator(company_name=company_name)
        integrated_report = aggregator.aggregate(claims, verification_response)
        
        print(f"💾 Saving integrated report...")
        aggregator.save_report(
            integrated_report,
            path=str(PROJECT_ROOT / "res" / "final_report.json")
        )
        
        # Print summary
        print(f"\n📈 Summary for {company_name}:")
        print(f"   Total Claims: {integrated_report.summary.total_claims}")
        print(f"   ✅ Supported: {integrated_report.summary.supported}")
        print(f"   ❌ Contradicted: {integrated_report.summary.contradicted}")
        print(f"   ⚠️  Insufficient Evidence: {integrated_report.summary.insufficient_evidence}")
        print(f"   🚨 High Risk: {integrated_report.summary.high_risk_count}")
        print(f"   Evidence: {integrated_report.summary.evidence_breakdown}")
        
        print(f"✅ Company {idx} completed!")
    
    print("\n" + "="*60)
    print("🎉 All processing completed!")
    print("="*60)
    print(f"\n📄 Final report saved to: res/final_report.json")
    

if __name__ == "__main__":
    main()