import io
import os
import uuid
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ai import model, calculate_overlaps, save_labeled_image, find_objects

app = FastAPI()

# Directory for labeled images
LABELS_DIR = "labeled_images"
os.makedirs(LABELS_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for chair occupancy statistics.
    """
    try:
        # Load the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Infer using the YOLO model
        results = model.predict(source=image, save=False)

        # Get statistics
        stats = calculate_overlaps(results)
        return JSONResponse(content={
            "filename": file.filename,
            "persons_detected": stats['person_count'],
            "chairs_detected": stats['chair_count'],
            "overlaps": stats['overlap_count'],
            "occupancy_percentage": stats['occupancy_percentage']
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/label-image")
async def label_image(file: UploadFile = File(...)):
    """
    Accepts an image, creates a labeled version with bounding boxes, and returns the labeled image.
    """
    try:
        # Save the uploaded image temporarily
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Run YOLO predictions and draw bounding boxes
        # results = model.predict(source=temp_path, save=False, conf=0.5)
        results = find_objects(temp_path)

        # Save the labeled image with bounding boxes
        labeled_image_path = save_labeled_image(temp_path, results, LABELS_DIR)

        # Return labeled image as a response
        with open(labeled_image_path, "rb") as f:
            return StreamingResponse(BytesIO(f.read()), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
