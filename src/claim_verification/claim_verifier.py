import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from claim_verification.search_manager import SearchManager
from common.schemes import VerificationList, ClaimsResponse

load_dotenv()

class ClaimVerifier():
    
    def __init__(self, api_key, model="gpt-5.2", temperature=0):
        """Class constructor

        Args:
            api_key (str): API_KEY to get access to the model
            model (str, optional): Model that the user wants to use. Defaults to "gpt-4o".
            temperature (int, optional): Degree of freedom for the model. Defaults to 0.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.search_manager = SearchManager() 
        
    def setup(self):
        """Perform setup of the model to verify claims
        """
        chat_model = ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
        structured_verification_llm = chat_model.with_structured_output(VerificationList)
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{search_results}")
        ])
        
        self.verification_chain = verification_prompt | structured_verification_llm
        self.search_manager.setup()
        self.is_setup = True
    
    def _get_system_prompt(self):
        """Return the system prompt for claim verification"""
        return """You are an expert claim verification analyst with deep expertise in due diligence and fact-checking.

Your task is to verify claims with rigorous attention to evidence quality and proper confidence calibration.

CRITICAL CONFIDENCE CALIBRATION RULES:

1. INSUFFICIENT_EVIDENCE verdicts MUST have LOW confidence (0.3-0.5 max):
   - If search was incomplete or misdirected → confidence ≤ 0.4
   - If no relevant sources found → confidence ≤ 0.3
   - Never give high confidence when you lack evidence to verify

2. SOURCE INDEPENDENCE requirement for high confidence:
   - confidence > 0.8 requires 2+ INDEPENDENT primary sources
   - If sources cite each other or share the same origin → cap confidence at 0.7
   - Press releases, company websites are NOT independent of company claims
   - Industry reports, regulatory filings, third-party research are independent

3. INTERNAL DOCUMENT references (pitch decks, product taglines, company materials):
   - Claims from internal documents CANNOT be externally verified
   - Mark as INSUFFICIENT_EVIDENCE with confidence ≤ 0.4
   - Explicitly note in reasoning: "Claim references internal company materials that cannot be independently verified"

VERDICT GUIDELINES:
- SUPPORTED: Multiple independent sources confirm the claim (confidence 0.7-0.95)
- CONTRADICTED: Credible sources directly contradict the claim (confidence based on source quality)
- INSUFFICIENT_EVIDENCE: Cannot verify due to lack of sources, incomplete search, or internal-only references

Be conservative. Overconfidence is worse than acknowledging uncertainty."""
    
    def build_prompt(self, claims):
        claims = claims.model_dump_json()
        
        claims = json.loads(claims)
        
        return f"""TASK:
Verify each of the provided claims with careful attention to confidence calibration.

CLAIMS:
{claims}

VERIFICATION RULES:

1. Use ONLY the sources provided for each claim.

2. CONFIDENCE CALIBRATION (CRITICAL):
   - INSUFFICIENT_EVIDENCE → confidence MUST be 0.3-0.5 (never higher)
   - Single source only → confidence ≤ 0.7
   - Sources not independent (cite each other) → confidence ≤ 0.7
   - Multiple independent sources agree → confidence 0.75-0.9
   - Reserve 0.9+ for claims with 3+ high-quality independent sources

3. INTERNAL DOCUMENT DETECTION:
   - If claim references pitch deck content, product taglines, internal metrics, or company-specific terminology
   - Mark as INSUFFICIENT_EVIDENCE with confidence ≤ 0.4
   - State in reasoning: "This claim references internal company materials and cannot be independently verified"

4. SOURCE INDEPENDENCE CHECK:
   - Before giving confidence > 0.7, verify sources are truly independent
   - Company press releases, company website, founder interviews = NOT independent
   - Regulatory filings, third-party research, established news with original reporting = independent

5. Do NOT guess financial numbers, dates, or proprietary information.

6. Do NOT fabricate sources. No sources = INSUFFICIENT_EVIDENCE (confidence 0.3).

7. When uncertain, always choose INSUFFICIENT_EVIDENCE over incorrect SUPPORTED.

Process each claim one by one with these calibration rules in mind.""".strip()
    
    def verify_claims(self, claims):
        """Performs claim verification

        Args:
            claims (ClaimList): list of the claims that need to be verified

        Returns:
            VerificationList: List with the verification results for each claim
        """
        if not self.is_setup:
            print(f"Please setup the Claim Verifier first")
            return None, None
        
        search_results_raw = self.search_manager.perform_search(claims)
        
        search_results_prompt = self.build_prompt(search_results_raw)
        
        verification_response = self.verification_chain.invoke({
            "search_results": search_results_prompt
        })
        
        # Return both verification and raw search results for quality analysis
        return verification_response, search_results_raw
    
    def load_claims(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            
        claims = ClaimsResponse.model_validate(data)
        
        return claims
    
    def store_as_json(self, verified_claims, path="res/verification_response.json"):
        """_summary_

        Args:
            verified_claims (_type_): _description_
            path (str, optional): _description_. Defaults to "verification_response.json".
        """
        verified_claims_json = verified_claims.model_dump_json()
        
        verified_claims_dictionary = json.loads(verified_claims_json)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(verified_claims_dictionary, f, indent=4)
    

def main():
    # Prepare the API keys
    api_key = os.getenv("OPENAI_API_KEY")
    claim_verifier = ClaimVerifier(api_key=api_key)
    
    # Setup the Verifier
    claim_verifier.setup()
    
    # Load extracted claims 
    claims = claim_verifier.load_claims(os.path.join(os.path.dirname(__file__), "res/claims.json"))
    
    # Run claim verification
    verification_results = claim_verifier.verify_claims(claims)
    
    # Store the results in the JSON
    claim_verifier.store_as_json(verification_results)


if __name__ == "__main__":
    main()