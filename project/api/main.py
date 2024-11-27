from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import torch
from .model_loader import load_model
from .utils import preprocess_image, decode_prediction
from fastapi.staticfiles import StaticFiles

app = FastAPI()
static_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_folder_path), name="static")

# 모델 로드
model = load_model()
model.eval()


@app.get("/")
async def root():
    return {"message": "Welcome to the CIFAR-10 Prediction API. Use the /predict/ endpoint to POST an image."}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # 업로드된 파일 처리
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # 이미지 전처리
        input_tensor = preprocess_image(image)
        
        # 모델 추론
        with torch.no_grad():
            outputs = model(input_tensor)
            result = decode_prediction(outputs)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

