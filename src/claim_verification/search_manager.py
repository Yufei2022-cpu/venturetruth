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
            max_tokens=1000,  # Increased for detailed per-source analysis
            timeout=90,
            model_kwargs={
                "extra_body": {
                    "return_related_questions": False,
                    "num_search_results": 8,  # More results to filter
                },
            }
        )
        
        structured_llm = chat_model.with_structured_output(SearchResults)
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_search_system_prompt()),
            ("human", "{claim_text}")
        ])
        
        self.verification_chain = verification_prompt | structured_llm
        
        self.is_setup = True
    
    def _get_search_system_prompt(self):
        """Return the system prompt for targeted search"""
        return """You are an expert research analyst performing targeted fact-checking searches.

SEARCH METHODOLOGY (FOLLOW STRICTLY):

STEP 1 - ENTITY RESOLUTION:
- Extract the company/entity name from the claim
- Identify any alternative names, parent companies, or related entities
- Use the CORRECT entity name in your search (avoid generic terms)

STEP 2 - TARGETED QUERY CONSTRUCTION:
- Build a specific search query combining: [Entity Name] + [Key Claim Keywords]
- Example: "Tesla 2023 vehicle deliveries quarterly report" NOT "electric car deliveries"
- Include relevant timeframes, metrics, or specific terms from the claim

STEP 3 - PER-SOURCE EVALUATION:
For EACH source found, you MUST provide:
- The full URL
- Relevance rating: HIGH (directly addresses claim), MEDIUM (related context), LOW (tangential), OFF_TOPIC (irrelevant)
- A specific excerpt if the source contains relevant information
- If no relevant info: explain WHY (e.g., "Article discusses different product line", "Data is from wrong time period")

STEP 4 - SEARCH QUALITY ASSESSMENT:
Rate your overall search:
- GOOD: Found 2+ HIGH relevance sources that directly address the claim
- PARTIAL: Found some relevant context but no direct confirmation/refutation
- FAILED: No relevant sources found, or all sources are OFF_TOPIC

CRITICAL RULES:
1. NEVER return off-topic sources without marking them as OFF_TOPIC
2. If claim references internal documents (pitch deck, internal metrics), note: "Internal document reference - cannot verify externally"
3. Include ONLY relevant source URLs in the 'sources' list (filter out OFF_TOPIC)
4. Document EXACTLY what you searched for in 'search_query'
5. If search returns irrelevant results, explain why in 'search_notes'
6. Keep search_results summary under 100 words

Your output enables downstream verification - be precise and auditable."""
    
    def perform_search(self, claims):
        if not self.is_setup:
            print(f"Please set up the Search Manager first!")
            return
        
        results = []
        total_claims = len(claims.claims)
        print(f"\n🔍 Starting targeted search for {total_claims} claims...")
        
        for idx, claim in enumerate(claims.claims, 1):
            claim_preview = claim.claim[:60] + "..." if len(claim.claim) > 60 else claim.claim
            print(f"  [{idx}/{total_claims}] Searching: {claim_preview}")
            
            try:
                verification_response = self.verification_chain.invoke({
                    "claim_text": f"""CLAIM TO VERIFY:
ID: {claim.id}
Claim: {claim.claim}
Original Evidence: {claim.evidence}

Perform a targeted search following the methodology. Document your search query, entity identified, and provide detailed per-source analysis."""
                })
                
                # Log search quality
                quality_icon = {"GOOD": "✅", "PARTIAL": "⚠️", "FAILED": "❌"}.get(verification_response.search_quality, "❓")
                relevant_count = len([s for s in verification_response.source_details if s.relevance in ["HIGH", "MEDIUM"]])
                print(f"  {quality_icon} {claim.id}: {verification_response.search_quality} ({relevant_count} relevant sources)")
                
                results.append(verification_response)
                
            except Exception as e:
                print(f"  ❌ Error searching {claim.id}: {str(e)}")
                # Create a failed search result
                from common.schemes import SourceDetail
                failed_result = SearchResults(
                    claim=claim,
                    search_query=f"Failed to construct query for: {claim.claim[:50]}",
                    entity_identified="Unknown",
                    search_results="Search failed due to technical error",
                    source_details=[],
                    sources=[],
                    search_quality="FAILED",
                    search_notes=f"Error: {str(e)}"
                )
                results.append(failed_result)
            
        search_results_list = SearchResultsList(search_results_list=results)
        
        # Print summary
        good_count = len([r for r in results if r.search_quality == "GOOD"])
        partial_count = len([r for r in results if r.search_quality == "PARTIAL"])
        failed_count = len([r for r in results if r.search_quality == "FAILED"])
        print(f"\n📊 Search Summary: ✅ {good_count} GOOD | ⚠️ {partial_count} PARTIAL | ❌ {failed_count} FAILED")
        
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