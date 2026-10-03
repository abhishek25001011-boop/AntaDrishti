"""Reusable YOLOv8 detection module for object detection."""

import cv2

from ultralytics import YOLO

from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD

TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}


def load_yolo_model():
    """Load the repository's YOLOv8n weights."""
    model_path = YOLO_MODEL_PATH
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"YOLO weights are missing or empty: {model_path}")
    return YOLO(str(model_path))


def run_detection(frame, model, conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD):
    """Run object detection on a single frame and return annotated output."""
    results = model(frame, device="cpu", conf=conf_threshold, verbose=False)
    output_frame = frame.copy()
    person_count = 0
    bag_count = 0
    detections = []

    if not results or len(results) == 0:
        return output_frame, 0, 0, []

    result = results[0]
    boxes = result.boxes
    names = model.names

    if boxes is None or len(boxes) == 0:
        return output_frame, 0, 0, []

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for coords, conf, cls_id in zip(xyxy, confidences, classes):
        label = names.get(cls_id, str(cls_id))
        if label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, coords)
        color = (0, 200, 0) if label == "person" else (0, 120, 255)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(
            output_frame,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detections.append({
            "label": label,
            "confidence": float(conf),
            "box": (x1, y1, x2, y2),
        })

        if label == "person":
            person_count += 1
        elif label in BAG_CLASSES:
            bag_count += 1

    return output_frame, person_count, bag_count, detections
