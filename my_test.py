# -*- coding:utf-8 -*-
# author:peng
# Date：2023/3/17 15:38
# detect.py
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)
print(sys.path)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
print(Path.cwd())
print(ROOT)