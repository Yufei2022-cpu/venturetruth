import os
import json

from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

from common.schemes import SearchResultsList, SearchResults, ClaimsResponse

load_dotenv()

api_key = os.getenv("PERPLEXITY_API_KEY")
    
class SearchManager:
    
    def __init__(self, api_key=api_key, model="sonar-pro", temperature=0, max_tokens=500, timeout=60):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.is_setup = False
    
    def setup(self):
        chat_model = ChatPerplexity(
            model="sonar-pro",
            temperature=0,
            pplx_api_key=api_key,
            max_tokens=500,
            timeout=60,
            model_kwargs={
                "extra_body": {
                    "return_related_questions": False,
                    "num_search_results" : 5,
                },
            }
        )
        
        structured_llm = chat_model.with_structured_output(SearchResults)
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a concise fact-checker. Verify the claim based on search results. "
                "CRITICAL INSTRUCTIONS: \n"
                "1. Keep the 'search_results' description UNDER 50 WORDS.\n"
                "2. In the 'sources' list, you MUST provide the FULL URL strings (starting with http/https).\n"
                "3. DO NOT use citation numbers like '[1]' in the 'sources' list.\n"
                "4. Be extremely brief.\n"
                "5. LIMIT the 'sources' list to a MAXIMUM of 5 URLs.\n"
                "6. The evidence string MUST be under 200 characters long. Summarize if necessary."
            )),
            ("human", "{claim_text}")
        ])
        
        self.verification_chain = verification_prompt | structured_llm
        
        self.is_setup = True
    
    def perform_search(self, claims):
        if not self.is_setup:
            print(f"Please set up the Search Manager first!")
            return
        
        results = []
        total_claims = len(claims.claims)
        print(f"\n🔍 Starting search for {total_claims} claims...")
        
        for idx, claim in enumerate(claims.claims, 1):
            print(f"  [{idx}/{total_claims}] Searching for claim {claim.id}: {claim.claim[:80]}...")
            
            verification_response = self.verification_chain.invoke({
                "claim_text": claim
            })
            
            results.append(verification_response)
            print(f"  ✓ Completed {claim.id}")
            
        search_results_list = SearchResultsList(search_results_list=results)
        print(f"✅ Search completed for all {total_claims} claims!\n")
        
        return search_results_list
    
    def load_claims(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            
        claims = ClaimsResponse.model_validate(data)
        
        return claims
    
    def store_as_json(self, search_results, path="res/search_results.json"):
        """Store provided search results as JSON

        Args:
            search_results : list of the search results for the claims
            path (str, optional): Path where to store the search results. Defaults to "search_results.json".
        """
        verified_search_results_json = search_results.model_dump_json()
        
        verified_search_results_dictionary = json.loads(verified_search_results_json)
        
        with open(path, "w") as f:
            json.dump(verified_search_results_dictionary, f, indent=4)

def main():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    search_manager = SearchManager(api_key=api_key)
    
    search_manager.setup()
    
    claims = search_manager.load_claims(os.path.join(os.path.dirname(__file__), "res/claims.json"))
    
    verification_results = search_manager.perform_search(claims)
    
    search_manager.store_as_json(verification_results)

if __name__ == "__main__":
    main()