import torch
from pathlib import Path
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(ROOT)

# Model
model = torch.hub.load('.', 'yolov5s',source='local',pretrained=True)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = './data/images/bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()
