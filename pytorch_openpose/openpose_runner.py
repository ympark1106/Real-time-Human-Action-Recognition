# openpose_runner.py
import sys
from pathlib import Path
OPENPOSE_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(OPENPOSE_ROOT))

from src.body import Body
from src.util import draw_bodypose

class OpenPoseRunner:
    def __init__(self, model_path):
        """
        OpenPose 모델 초기화
        """
        
        self.model = Body(str(model_path))

    def process(self, crop_img):
        """
        OpenPose로 Skeleton 추출
        """
        candidate, subset = self.model(crop_img)
        canvas = draw_bodypose(crop_img.copy(), candidate, subset)
        return canvas, candidate, subset
