import opendatasets as od
import pandas

od.download(
    "https://www.kaggle.com/competitions/mc-datathon-2025-players-detection/data")
# 🧠 Install & setup
!pip install ultralytics kaggle roboflow opencv-python-headless
import os
from ultralytics import YOLO
import pandas as pd
import cv2

# 🔑 Download Kaggle dataset (ensure kaggle.json is configured in /root/.kaggle/)
!kaggle competitions download -c mc-datathon-2025-football-players-detection -p ./kaggle_data
!unzip -q ./kaggle_data/*.zip -d ./kaggle_data

# 🌐 Download Roboflow augmentation dataset (optional)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")
project = rf.workspace().project("football-players-detection-3zvbc")
dataset = project.version(2).download("yolov8")  # Adjust version as needed

# ➕ Merge Kaggle + Roboflow data
os.system(f"rm -rf merged; mkdir merged")
os.system("cp -r kaggle_data/train merged/")
os.system(f"cp -r {dataset.location}/train/* merged/train/")
# similarly for 'valid' folders

# 🔧 Create YOLOv8 config YAML (-- you can customize)
with open("data.yaml","w") as f:
    f.write("""
train: merged/train/images
val:   merged/valid/images
nc: 1
names: ['player']
""")

# 🚀 Train YOLOv8 (using medium model, tune epochs and imgsz as needed)
model = YOLO("yolov8m.pt")
metrics = model.train(data="data.yaml", epochs=100, imgsz=640, batch=16, patience=10)

# 📦 Inference on test set
preds = model.predict(source="kaggle_data/test/images", imgsz=640, conf=0.25, save=False)

# 📝 Format submission file
out = []
for r in preds:
    fname = os.path.basename(r.path)
    boxes = r.boxes.xyxy.tolist()  # list of [x1, y1, x2, y2]
    for b in boxes:
        x1, y1, x2, y2 = b
        out.append({
            "image_id": fname,
            "xmin": int(x1), "ymin": int(y1),
            "xmax": int(x2), "ymax": int(y2),
            "confidence": float(r.boxes.conf[0]) if hasattr(r.boxes, 'conf') else 1.0,
            "class": 0
        })

df = pd.DataFrame(out)
df.to_csv("submission.csv", index=False)
print("✅ submission.csv generated!")
