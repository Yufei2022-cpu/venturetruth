from claim import Claim

from pydantic import BaseModel, Field

class VerificationResult(BaseModel):
    claim: Claim = Field(description="Claim")
    valid: bool = Field(description="Indicator whether the claim is valid")
    reasoning: str = Field(description="Explanation why the calim is true/false")
    sources: list[str] = Field(description="List of the URL sources used to support the reasoning")