import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/fracture_detector_final.pth"
    CLASS_NAMES: list = ['not_fractured', 'fractured']
    
    IMAGE_SIZE: int = 224
    IMG_MEAN: list = [0.485, 0.456, 0.406]
    IMG_STD: list = [0.229, 0.224, 0.225]
    
    CONFIDENCE_THRESHOLD: float = 0.80
    
    class Config:
        extra = "ignore"
        
settings = Settings()