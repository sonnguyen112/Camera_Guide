from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import base64
import numpy as np
from typing import Any
from utils import predict_align_angle

app = FastAPI()

# Define the request body schema


class ImageBase64Request(BaseModel):
    img_base64: str


@app.get("/")
async def hello():
    return {"hello": "world"}


@app.post("/process-image")
async def process_image(request: ImageBase64Request) -> Any:
    try:
        # Decode the Base64 image string
        # print(request.img_base64)
        img_bytes = base64.b64decode(request.img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # Convert the NumPy array to an OpenCV image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Rotate 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        angle, _ = predict_align_angle(img)

        return {"status": "success", "rotation_angle": angle}

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=400, detail=f"Image processing failed: {e}")


@app.post("/rotate-image")
async def rotate_img(request: ImageBase64Request):
    try:
        # Decode the Base64 image string
        # print(request.img_base64)
        img_bytes = base64.b64decode(request.img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # Convert the NumPy array to an OpenCV image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Rotate 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        _, rotated_img = predict_align_angle(img, reduce_size=False)

        rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to an base64-encoded string
        rotated_img_bytes = cv2.imencode(".jpg", rotated_img)[1].tobytes()
        rotated_img_base64 = base64.b64encode(
            rotated_img_bytes).decode("utf-8")
        
        rotated_img_base64 = "data:image/jpeg;base64," + rotated_img_base64

        return {"status": "success", "rotated_image": rotated_img_base64}

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=400, detail=f"Image processing failed: {e}")
