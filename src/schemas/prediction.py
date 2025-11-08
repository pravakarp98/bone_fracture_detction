from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    error: str | None = None