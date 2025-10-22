"""
import os

image_dir = "labeled_pieces"
label_dir = "labels"
os.makedirs(label_dir, exist_ok=True)

classes = ['e', 'h', 'wp', 'wk', 'wb', 'wr', 'wq', 'wn', 'bn', 'bp', 'bk', 'bb', 'br', 'bq']
label_map = {cls: idx for idx, cls in enumerate(classes)}

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        for cls in classes:
            if f"_{cls}" in filename:
                class_id = label_map[cls]
                label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                break
"""

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="chess.yaml", epochs=50, imgsz=150)
