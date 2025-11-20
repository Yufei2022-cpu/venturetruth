from pydantic import BaseModel, Field

from claim import Claim

class ClaimList(BaseModel):
    claims: list[Claim] = Field(description="Claims extracted from the file")