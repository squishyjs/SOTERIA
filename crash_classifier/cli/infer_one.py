from ultralytics import YOLO
from PIL import Image
import sys, torch

model = YOLO("runs/classify/exp/weights/best.pt")   # adjust path if needed
img = Image.open(sys.argv[1]).convert("RGB")
prob = model(img)[0].probs

print(f"Prediction: {'crash' if prob.top1 == 0 else 'normal'}  "
      f"(p={prob.softmax().max():.2f})")
