from verification_result import VerificationResult

from pydantic import BaseModel, Field

class VerificationList(BaseModel):
    verification_results: list[VerificationResult] = Field(description="List of the verification results")