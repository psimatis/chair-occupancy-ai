import os
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO

model_path = "models/yolo11x.pt"
model = YOLO(model_path)

RELEVANT_CLASS_IDS = [0, 13, 56, 57, 59] # Person, Bench, Chair, Couch, Bed

def find_objects(image):
    """Run YOLO with restricted results to releevant classes."""
    return model.predict(source=image, classes=RELEVANT_CLASS_IDS, save=False, conf=0.23)

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0

def calculate_stats(results, iou_threshold=0.01):
    """Calculate overlaps, utilization details, and person-to-chair ratio."""
    stats = defaultdict(int)
    person_boxes = []
    chair_boxes = []

    # Extract bounding boxes for relevant classes
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == RELEVANT_CLASS_IDS[0]:
                person_boxes.append(box.tolist())
            elif int(cls) in RELEVANT_CLASS_IDS[1:]:
                chair_boxes.append(box.tolist())

    # Calculate occupied chairs
    matched_chairs = set()
    for person_box in person_boxes:
        for chair_box in chair_boxes:
            if tuple(chair_box) in matched_chairs:
                continue  # Skip occupied chairs

            if compute_iou(person_box, chair_box) >= iou_threshold:
                stats['chairs_taken'] += 1
                matched_chairs.add(tuple(chair_box))  
                break  # Exit the loop after finding a match for this person

    stats['chairs_taken'] = min(stats['chairs_taken'], len(chair_boxes), len(person_boxes))

    # Chair utilization details
    stats['chairs'] = len(chair_boxes)
    stats['people'] = len(person_boxes)
    stats['empty_chairs'] = stats['chairs'] - stats['chairs_taken']
    stats['min_occupancy'] = (stats['chairs_taken'] / stats['chairs']) * 100 if stats['chairs'] > 0 else 0
    stats['max_occupancy'] = (stats['people'] / stats['chairs']) * 100 if stats['chairs'] > 0 else 0
    print('Stats:', stats)
    return stats

def save_labeled_image(input_image, results, output_dir):
    """Create and save labeled images."""
    for result in results:
        labeled_image = result.plot() 

    # Convert to PIL and from BGR to RGB
    labeled_image = Image.fromarray(labeled_image[:, :, ::-1])  

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    labeled_image_path = os.path.join(output_dir, f"labeled_{os.path.basename(input_image)}")
    labeled_image.save(labeled_image_path)
    return labeled_image_path

