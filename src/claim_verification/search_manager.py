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
            max_tokens=2000,  # Increased for detailed per-source analysis with more sources
            timeout=120,
            model_kwargs={
                "extra_body": {
                    "return_related_questions": False,
                    "num_search_results": 20,  # Maximized results for comprehensive coverage
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
        return """You are an expert research analyst performing comprehensive fact-checking searches.

YOUR PRIMARY GOAL: Find AS MANY RELEVANT SOURCES AS POSSIBLE. The more high-quality sources you find, the better the verification will be.

SEARCH METHODOLOGY (FOLLOW STRICTLY):

STEP 1 - ENTITY RESOLUTION:
- Extract the company/entity name from the claim
- Identify any alternative names, parent companies, subsidiaries, or related entities
- Look for the entity in multiple contexts (news, official sources, industry reports, regulatory filings)

STEP 2 - MULTI-ANGLE QUERY CONSTRUCTION:
- Construct MULTIPLE search queries from different angles to maximize source coverage:
  a) Official sources: "[Entity] official announcement/press release"
  b) News coverage: "[Entity] [claim keywords] news"
  c) Financial/regulatory: "[Entity] SEC filing/annual report/earnings"
  d) Industry analysis: "[Entity] market research/industry report"
  e) Third-party verification: "[Entity] [claim keywords] verified/confirmed"
- Example: For "Tesla delivered 1.8M vehicles in 2023"
  -> "Tesla 2023 vehicle deliveries official", "Tesla annual report 2023", "Tesla Q4 2023 earnings deliveries"

STEP 3 - MAXIMIZE SOURCE COLLECTION:
- IMPORTANT: Aim to find AT LEAST 5-10 DIFFERENT sources for each claim
- Prioritize diverse source types: news outlets, official company sources, regulatory filings, analyst reports, industry publications
- Include sources from different dates to show consistency of information
- Do NOT limit yourself to the first few results - dig deeper

STEP 4 - PER-SOURCE EVALUATION:
For EACH source found, you MUST provide:
- The full URL
- Relevance rating: HIGH (directly addresses claim), MEDIUM (related context), LOW (tangential), OFF_TOPIC (irrelevant)
- A specific excerpt if the source contains relevant information
- If no relevant info: explain WHY (e.g., "Article discusses different product line", "Data is from wrong time period")

STEP 5 - SEARCH QUALITY ASSESSMENT:
Rate your overall search:
- EXCELLENT: Found 5+ HIGH relevance sources with diverse perspectives
- GOOD: Found 3-4 HIGH relevance sources that directly address the claim
- PARTIAL: Found some relevant context but fewer than 3 direct sources
- FAILED: No relevant sources found, or all sources are OFF_TOPIC

CRITICAL RULES:
1. NEVER stop searching after finding just 1-2 sources - always try to find more
2. ALWAYS include the full list of ALL sources found, even if some are MEDIUM or LOW relevance
3. If claim references internal documents (pitch deck, internal metrics), note: "Internal document reference - cannot verify externally" BUT STILL search for any public corroboration
4. Document ALL search queries you tried in 'search_query' (comma-separated)
5. Include cross-referencing sources that confirm the same information from different angles
6. For financial claims, ALWAYS look for SEC filings, earnings reports, and analyst coverage

Your output enables downstream verification - the MORE sources you find, the more confident the verification will be. Be comprehensive and thorough."""
    
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