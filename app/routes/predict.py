from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
from pydantic import BaseModel
from app.services.model_loader import predict_image_from_url, predict_image_from_file
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

router = APIRouter()

# Dependency to check API key from header
def check_api_key(req_api_key: str = Header(...)):
    if req_api_key != api_key:
        raise HTTPException(status_code=403, detail="Unauthorized")

class ImageURL(BaseModel):
    image_url: str


@router.post("/url", dependencies=[Depends(check_api_key)])
async def predict_from_url(data: ImageURL):
    try:
        result = predict_image_from_url(data.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Secured endpoint for file prediction
@router.post("/file", dependencies=[Depends(check_api_key)])
async def predict_from_file(file: UploadFile = File(...)):
    try:
        result = predict_image_from_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
