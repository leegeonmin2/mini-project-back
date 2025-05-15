from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

@app.post("/api/analyze-photo")
async def analyze_photo(image: UploadFile = File(...)):
    # 이미지 저장
    image_path = f"temp_{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 분석 함수 호출
    from model import analyze_face
    result = analyze_face(image_path)

    # 임시 이미지 삭제
    os.remove(image_path)

    return JSONResponse(content=result)
