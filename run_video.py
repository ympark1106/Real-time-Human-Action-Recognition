import os
import cv2
from pathlib import Path
import numpy as np
import sys
import argparse
sys.path.append(str('C:/Users/USER/Workspace/HAL/'))

from yolov9.yolo_runner import run_yolo
from pytorch_openpose.openpose_runner import load_openpose_model, process_with_openpose
from sort.sort import Sort

# 경로 설정
YOLO_WEIGHTS = 'yolov9/weights/gelan-c-det.pt'

def main(args):
    # GPU 설정
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU ID: {args.gpu_id}")
    else:
        print("Using default GPU (or CPU if no GPU available).")

    # OpenPose 모델 로드
    openpose_model = load_openpose_model()

    # SORT 객체 추적 초기화
    tracker = Sort()

    # 비디오 캡처 초기화
    if not os.path.exists(args.input):
        print(f"Error: Input video {args.input} does not exist.")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.input}")
        return

    # 비디오 저장 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 실행
        person_list, im0 = run_yolo(frame, YOLO_WEIGHTS)

        # SORT 입력 준비 (Bounding Box와 신뢰도)
        detections = []
        for xyxy, crop_img in person_list:
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = 5.0  # YOLO에서 가져온 신뢰도 추가 가능
            detections.append([x1, y1, x2, y2, confidence])

        # SORT로 Tracking
        tracked_objects = tracker.update(np.array(detections))

        # Bounding Box 및 ID 표시
        for x1, y1, x2, y2, obj_id in tracked_objects:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"Person {int(obj_id)}"
            color = (0, 255, 0)  # 초록색

            # Bounding Box 및 라벨 추가
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # OpenPose Skeleton 추가
            crop_img = im0[y1:y2, x1:x2].copy()
            pose_canvas, _, _ = process_with_openpose(openpose_model, crop_img)

            # 크기 맞추기
            pose_canvas = cv2.resize(pose_canvas, (x2 - x1, y2 - y1))

            # 채널 수 맞추기
            if len(pose_canvas.shape) == 2:
                pose_canvas = cv2.cvtColor(pose_canvas, cv2.COLOR_GRAY2BGR)

            # 원본 이미지에 Skeleton 덮어쓰기
            im0[y1:y2, x1:x2] = cv2.addWeighted(im0[y1:y2, x1:x2], 0.5, pose_canvas, 0.5, 0)

        # 결과 저장
        out.write(im0)

        # 결과 화면 출력
        cv2.imshow("YOLO + OpenPose + Tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + OpenPose + Tracking Video Processor")
    parser.add_argument("-g","--gpu_id", type=int, default=0, help="GPU ID to use (default: None for auto-selection)")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="output_with_skeleton.mp4", help="Path to save the output video")

    args = parser.parse_args()
    main(args)