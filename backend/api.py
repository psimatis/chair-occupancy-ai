import os
import uuid
import time
from io import BytesIO
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from yolo import calculate_stats, save_labeled_image, find_objects
from gemini import analyze_image as gemini_analyze

app = FastAPI()

# Directory for labeled images
LABELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../labeled_images"))
os.makedirs(LABELS_DIR, exist_ok=True)
app.mount("/labeled_images", StaticFiles(directory=LABELS_DIR), name="labeled_images")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyzes image for chair occupancy statistics and returns labeled image."""
    try:
        start_time = time.time()

        # Load the image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        image.save(temp_path)

        # Process the image
        model_start_time = time.time()
        results = find_objects(image)
        model_time = time.time() - model_start_time
        stats = calculate_stats(results)

        # Save the labeled image
        labeled_image_path = save_labeled_image(temp_path, results, LABELS_DIR)

        # Encode the labeled image as Base64
        with open(labeled_image_path, "rb") as labeled_image_file:
            labeled_image_base64 = base64.b64encode(labeled_image_file.read()).decode("utf-8")

        total_api_time = time.time() - start_time

        return JSONResponse(content={
            "filename": file.filename,
            "people": stats['people'],
            "chairs": stats['chairs'],
            "chairs_taken": stats['chairs_taken'],
            "empty_chairs": stats['empty_chairs'],
            "min_occupancy": stats['min_occupancy'],
            "max_occupancy": stats['max_occupancy'],
            "labeled_image_base64": labeled_image_base64,
            "timing": {
                "model_prediction_time": model_time,
                "total_api_time": total_api_time
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/llm-analyze")
async def llm_analyze_image(file: UploadFile = File(...)):
    """Uses Gemini API to return a user-frieldly analysis."""
    try:
        # Save the uploaded image temporarily
        image_data = await file.read()
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(image_data)

        # Call the Gemini API
        gemini_result = gemini_analyze(temp_path)

        # Remove the temporary file
        os.remove(temp_path)

        return JSONResponse(content={
            "filename": file.filename,
            "gemini_analysis": gemini_result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))