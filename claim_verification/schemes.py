from pydantic import BaseModel, Field

class Claim(BaseModel):
    id: str = Field(description="Id of the claim.")
    claim: str = Field(description="Claim to be verified.")
    confidence: float = Field(description="Preliminary confidence that the claim is valid. It does not mean that the claim is valid.")
    evidence: str = Field(description="Paragraph name where the claim was found. It cannot be used to verify the claim.")

class ClaimList(BaseModel):
    claims: list[Claim] = Field(description="Claims extracted from the file")
    
class SearchResults(BaseModel):
    claim: Claim = Field(description="Claim for which the search results are provided")
    search_results: str = Field(description="Results of the internet search")
    sources: list[str] = Field(description="Sources for the provided search results")
    
class SearchResultsList(BaseModel):
    search_results_list: list[SearchResults] = Field(description="List of all search results")
    
class VerificationResult(BaseModel):
    claim: Claim = Field(description="Claim")
    valid: bool = Field(description="Indicator whether the claim is valid")
    reasoning: str = Field(description="Explanation why the calim is true/false")
    sources: list[str] = Field(description="List of the URL sources used to support the reasoning")
    
class VerificationList(BaseModel):
    verification_results: list[VerificationResult] = Field(description="List of the verification results")