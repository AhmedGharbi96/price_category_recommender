from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    class Config:
        # re-validate model when edited
        validate_assignment = True
