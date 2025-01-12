# yolo_runner.py
import sys
from pathlib import Path
import cv2
import torch
import numpy as np

YOLO_ROOT = Path(__file__).resolve().parents[0] 
sys.path.append(str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YoloRunner:
    def __init__(self, weights, conf_thres=0.5, iou_thres=0.5, img_size=640):
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size(img_size, s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self, frame):
        """
        YOLO 실행 함수: 단일 프레임 처리
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Error: Input frame is not a valid numpy array.")
        
        # YOLO 입력에 맞게 전처리
        frame_resized = cv2.resize(frame, (self.imgsz, self.imgsz))  # YOLO 입력 크기로 리사이즈
        frame_rgb = frame_resized[:, :, ::-1].transpose(2, 0, 1)  # HWC -> CHW 형식으로 변환
        frame_normalized = np.ascontiguousarray(frame_rgb) / 255.0  # 0-255 -> 0-1 정규화

        # 배치 차원 추가
        im = torch.from_numpy(frame_normalized).to(self.device).float()
        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # 배치 차원 추가 (1, 3, H, W)

        # YOLO 추론
        pred = self.model(im)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # 감지 결과 처리
        person_list = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    if self.names[int(cls)] == 'person':
                        crop_img = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        person_list.append((xyxy, crop_img))

        return person_list, frame