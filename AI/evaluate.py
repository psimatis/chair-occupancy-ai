import os
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch

data_yaml = {
    "train": "Datasets/D1/train/images",
    "val": "Datasets/D1/valid/images",
    "test": "Datasets/D1/test/images",
    "ground_truth_dir": "Datasets/D1/test/labels",
}
model_path = "models/yolov8x.pt"
project_dir = "/home/panos/Desktop/chair-occupancy-ai/runs"  # Directory for predictions

# Necessary classes
person_class_id = 0  # Person class
chair_class_ids = [13, 56, 57, 59]  # Bench, Chair, Couch, Bed

model = YOLO(model_path)

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    """
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
    """
    Filter predictions to include only person and chair-related classes.
    """
    filtered_data = []

    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i]
        score = result.boxes.conf[i]
        cls = result.boxes.cls[i]
        if int(cls) == person_class_id or int(cls) in chair_class_ids:  # Filtering
            filtered_data.append(torch.cat([box, torch.tensor([score, cls])]))

    if filtered_data:
        relevant_boxes = torch.stack(filtered_data)
        result.boxes = Boxes(relevant_boxes, orig_shape=result.orig_shape)
    else:
        # Create an empty Boxes object if no data matches
        result.boxes = Boxes(torch.empty((0, 6)), orig_shape=result.orig_shape)

def calculate_overlaps(results, iou_threshold=0.3):
    """
    Calculate overlaps between "person" and "chair" bounding boxes (i.e., occupied chairs)
    """
    person_boxes = []
    chair_boxes = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == person_class_id:
                person_boxes.append(box.tolist())
            elif int(cls) in chair_class_ids: 
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

def evaluate_model():
    """
    Evaluate by calculating overlaps between "person" and "chair" bounding boxes per image.
    """
    test_images_dir = data_yaml["test"]
    output_dir = os.path.join(project_dir, "filtered_images")

    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_file)
        results = model.predict(source=image_path, save=False)

        # Keep only person and chair-related classes
        for result in results:
            filter_classes(result)

        # Save images
        output_image_path = os.path.join(output_dir, os.path.basename(image_file))
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        result.plot(save=True, filename=output_image_path)

        # Calculate overlaps
        stats = calculate_overlaps(results)

        # Results
        print(f"Image: {image_file}")
        print(f"  Persons Detected: {stats['person_count']}")
        print(f"  Chairs Detected: {stats['chair_count']}")
        print(f"  Overlaps (Person-Chair): {stats['overlap_count']}")
        print(f"  Chair Occupancy Percentage: {stats['occupancy_percentage']:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    evaluate_model()
