import torch
from torchvision import transforms
from PIL import Image
from src.config import settings

image_transforms = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.IMG_MEAN, std=settings.IMG_STD)
])

def predict_image(image, model, device):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_tensor = image_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction_index = torch.argmax(probabilities, dim=1).item()
        
    confidence = probabilities[0][prediction_index].item()
    prediction_name = settings.CLASS_NAMES[prediction_index]
    
    if confidence < settings.CONFIDENCE_THRESHOLD:
        return "Unknown", 0.0, "Prediction failed: Image may not be a valid X-ray. Confidence is too low."
    
    return prediction_name, confidence, None