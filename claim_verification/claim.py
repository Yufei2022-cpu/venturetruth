from pydantic import BaseModel, Field

class Claim(BaseModel):
    id: str = Field(description="Id of the claim")
    claim: str = Field(description="Claim to be verified")
    support: str = Field(description="Can be either dirent, indiredt, or uncertain")
    source_fragment: str = Field(description="Source, where the claim was found")