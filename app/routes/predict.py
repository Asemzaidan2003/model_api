from pydantic import BaseModel
from app.services.model_loader import predict_image_from_url, predict_image_from_file
import os
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header

load_dotenv()
api_key = os.getenv("API_KEY")

router = APIRouter()

# Dependency to check API key from header
def check_api_key(req_api_key: str = Header(...)):
    print("check_api_key")
    if req_api_key is None:
        print("API key is None")
        raise HTTPException(status_code=403, detail="no key !")
    if req_api_key != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

class ImageURL(BaseModel):
    image_url: str


@router.post("/url", dependencies=[Depends(check_api_key)])
async def predict_from_url(data: ImageURL):
    print("predict_from_url_route")
    try:
        result = predict_image_from_url(data.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file", dependencies=[Depends(check_api_key)])
async def predict_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_image_from_file(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))