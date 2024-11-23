import io
import os
import uuid
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ai import calculate_overlaps, save_labeled_image, find_objects


app = FastAPI()

# Directory for labeled images, stored outside the source folder
LABELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../labeled_images"))
os.makedirs(LABELS_DIR, exist_ok=True)

# Mount the labeled images directory as a static file route
app = FastAPI()
app.mount("/labeled_images", StaticFiles(directory=LABELS_DIR), name="labeled_images")


@app.post("/analyze-and-label")
async def analyze_and_label_image(file: UploadFile = File(...)):
    """
    Accepts an image, analyzes it for chair occupancy statistics,
    saves a labeled version of the image, and returns statistics and the labeled image path.
    """
    try:
        # Load the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Save the uploaded image temporarily
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        image.save(temp_path)

        # Infer using the YOLO model
        results = find_objects(image)

        # Get statistics
        stats = calculate_overlaps(results)

        # Save the labeled image with bounding boxes
        labeled_image_path = save_labeled_image(temp_path, results, LABELS_DIR)

        # Return statistics and labeled image path
        return JSONResponse(content={
            "filename": file.filename,
            "labeled_image_path": labeled_image_path,
            "persons_detected": stats['person_count'],
            "chairs_detected": stats['chair_count'],
            "overlaps": stats['overlap_count'],
            "occupancy_percentage": stats['occupancy_percentage']
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)