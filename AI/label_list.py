from ultralytics import YOLO

# Load your YOLO model
model_path = "AI/yolo11x.pt"  # Path to your YOLO model
model = YOLO(model_path)

def get_yolo_labels(model):
    """
    Extract and print all the class labels from a YOLO model.
    """
    if hasattr(model, "names"):
        labels = model.names  # YOLO models store labels in the 'names' attribute
        return labels
    else:
        raise AttributeError("The YOLO model does not have a 'names' attribute.")

if __name__ == "__main__":
    labels = get_yolo_labels(model)
    print("YOLO Model Labels:")
    for class_id, label in labels.items():
        print(f"{class_id}: {label}")