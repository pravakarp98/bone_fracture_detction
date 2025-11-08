import torch
import torch.nn as nn
import torchvision
import logging
from src.config import settings

logger = logging.getLogger("uvicorn")

class ModdelLoader:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {self.device}")
        
        try:
            self.model = torchvision.models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.model.eval().to(self.device)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
    def get_model(self):
        return self.model, self.device
    
model_loader = ModdelLoader(settings.MODEL_PATH)