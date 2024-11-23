import os
import uuid
from io import BytesIO
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ai import calculate_overlaps, save_labeled_image, find_objects

app = FastAPI()

# Directory for labeled images
LABELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../labeled_images"))
os.makedirs(LABELS_DIR, exist_ok=True)
app.mount("/labeled_images", StaticFiles(directory=LABELS_DIR), name="labeled_images")

@app.post("/analyze-image")
async def analyze_and_label_json_image(file: UploadFile = File(...)):
    """
    Analyzes image for chair occupancy statistics and returns labeled image.
    """
    try:
        # Load the image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        image.save(temp_path)

        # Process the image
        results = find_objects(image)
        stats = calculate_overlaps(results)

        # Save the labeled image
        labeled_image_path = save_labeled_image(temp_path, results, LABELS_DIR)

        # Encode the labeled image as Base64
        with open(labeled_image_path, "rb") as labeled_image_file:
            labeled_image_base64 = base64.b64encode(labeled_image_file.read()).decode("utf-8")

        return JSONResponse(content={
            "filename": file.filename,
            "persons_detected": stats['person_count'],
            "chairs_detected": stats['chair_count'],
            "overlaps": stats['overlap_count'],
            "occupancy_percentage": stats['occupancy_percentage'],
            "labeled_image_base64": labeled_image_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)