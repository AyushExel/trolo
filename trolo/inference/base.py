from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any

import torch
from PIL import Image
import cv2

class BasePredictor(ABC):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    @abstractmethod
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from path"""
        pass
        
    @abstractmethod
    def preprocess(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess inputs to model input format"""
        pass
        
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Convert model outputs to final predictions"""
        pass

    def predict(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]]) -> Dict[str, Any]:
        """Run inference on inputs"""
        with torch.no_grad():
            preprocessed = self.preprocess(inputs)
            outputs = self.model(preprocessed)
            predictions = self.postprocess(outputs)
        return predictions 