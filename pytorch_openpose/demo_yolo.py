import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
import os

from src import model
from src import util
from src.body import Body
from src.hand import Hand

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def detect_pose_buffer(model, ori_img):
    candidate, subset = model(ori_img)
    canvas = copy.deepcopy(ori_img)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    return canvas

def load_pose_model():
    print(f"Loading pose model from {ROOT/'model/body_pose_model.pth'}")
    body_estimation = Body(ROOT/'model/body_pose_model.pth')
    return body_estimation




