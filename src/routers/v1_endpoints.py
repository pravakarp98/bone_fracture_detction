import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from src.schemas.prediction import PredictionResponse
from src.core.model_loader import model_loader
from src.core.inference import predict_image

router = APIRouter(
    prefix="/api/v1",
    tags=["v1_predctions"]
)

@router.post("/predict", response_model=PredictionResponse)
async def predict_fracture(file: UploadFile = File(...)):
    """
    Receives a X-ray image, runs the level 0 classfier, and returns whther it is 'fractured' or 'not_fractured'.

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        model, device = model_loader.get_model()
        
        prediction, confidence, error = predict_image(image, model, device)
        
        if error:
            return PredictionResponse(
                prediction="Error",
                confidence="0.00%",
                error=error
            )
            
        return PredictionResponse(
            prediction=prediction,
            confidence=f"{confidence * 100:.2f}%",
            error=None
        )
        
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not be opened.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occured: {str(e)}")