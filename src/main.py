import yaml
import os

from dotenv import load_dotenv

from claim_verification.claim_verifier import ClaimVerifier
from claim_extractor.claim_extractor import ClaimExtractor
from file_content_extraction.ingestion_pipeline import IngestionPipeline
from file_content_extraction.data_loader import DataLoader
from utils.utils import save_output

CONFIG_PATH = "config/config.yaml"

load_dotenv()

def load_configuration():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    return config
    

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    config = load_configuration()
    
    ingestion_pipeline = IngestionPipeline(
        csv_path=config['file_content_extractor']['csv_path'],
        pdf_folder=config['file_content_extractor']['pdf_folder'],
        output_path=config['file_content_extractor']['output_path'],
        limit=config['file_content_extractor']['limit']
    )
    
    data_loader = DataLoader(
        json_path=config['file_content_extractor']['output_path']
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
    ingestion_pipeline.run()
    
    # Load the Information & perform claim extraction and verificcation
    for _, item in enumerate(data_loader):
        claims = claim_extractor.extract_claims(item)

        print(claims)
        
        save_output(claims)
        
        verification_response = claim_verifier.verify_claims(claims)
        
        claim_verifier.store_as_json(verification_response)
    

if __name__ == "__main__":
    main()