from collections import defaultdict
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

model_path = "model/yolov8x.pt"
model = YOLO(model_path)

# Define classes
PERSON_CLASS_ID = 0
CHAIR_CLASS_IDS = [13, 56, 57, 59]

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

def filter_classes(result):
    """Filter predictions to include only person and chair-related classes."""
    filtered_data = []
    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i]
        score = result.boxes.conf[i]
        cls = result.boxes.cls[i]
        if int(cls) == PERSON_CLASS_ID or int(cls) in CHAIR_CLASS_IDS:
            filtered_data.append(torch.cat([box, torch.tensor([score, cls])]))

    if filtered_data:
        relevant_boxes = torch.stack(filtered_data)
        result.boxes = Boxes(relevant_boxes, orig_shape=result.orig_shape)
    else:
        result.boxes = Boxes(torch.empty((0, 6)), orig_shape=result.orig_shape)

def calculate_overlaps(results, iou_threshold=0.3):
    """Calculate overlaps between person and chair bounding boxes."""
    person_boxes = []
    chair_boxes = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == PERSON_CLASS_ID:
                person_boxes.append(box.tolist())
            elif int(cls) in CHAIR_CLASS_IDS:
                chair_boxes.append(box.tolist())

    stats = defaultdict(int)

    for person_box in person_boxes:
        for chair_box in chair_boxes:
            iou = compute_iou(person_box, chair_box)
            if iou >= iou_threshold:
                stats['overlap_count'] += 1

    stats['chair_count'] = len(chair_boxes)
    stats['person_count'] = len(person_boxes)
    stats['occupancy_percentage'] = (stats['overlap_count'] / stats['chair_count']) * 100 if stats['chair_count'] > 0 else 0
    return stats
