from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Render!"}
app.include_router(predict_router, prefix="/predict")