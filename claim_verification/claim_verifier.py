import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from search_manager import SearchManager
from schemes import VerificationList
from schemes import ClaimList

load_dotenv()

class ClaimVerifier():
    
    def __init__(self, api_key, model="gpt-4o", temperature=0):
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
            ("system", "Verify each of the provided quotes. For the verification only use the information provided in the 'search_results'."),
            ("human", "{search_results}")
        ])
        
        self.verification_chain = verification_prompt |structured_verification_llm
        self.search_manager.setup()
        self.is_setup = True
    
    def verify_claims(self, claims):
        """Performs claim verification

        Args:
            claims (ClaimList): list of the claims that need to be verified

        Returns:
            VerificationList: List with the verification results for each claim
        """
        if not self.is_setup:
            print(f"Please setup the Claim Verifier first")
            return
        
        search_results = self.search_manager.perform_search(claims)
        
        verification_response = self.verification_chain.invoke({
            "search_results": search_results
        })
        
        return verification_response
    
    def load_claims(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            
        claims = ClaimList.model_validate(data)
        
        return claims
    
    def store_as_json(self, verified_claims, path="res/verification_response.json"):
        """_summary_

        Args:
            verified_claims (_type_): _description_
            path (str, optional): _description_. Defaults to "verification_response.json".
        """
        verified_claims_json = verified_claims.model_dump_json()
        
        verified_claims_dictionary = json.loads(verified_claims_json)
        
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