import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ai import model, filter_classes, calculate_overlaps

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for chair occupancy statistics.
    """
    try:
        # Load the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Infer
        results = model.predict(source=image, save=False)

        # Filter irrelevant classes
        for result in results:
            filter_classes(result)

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
