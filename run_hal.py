import os
import cv2
import time
import numpy as np
import sys
import argparse
sys.path.append(str('C:/Users/USER/Workspace/HAL/'))

from yolov9.yolo_runner import run_yolo
from pytorch_openpose.openpose_runner import load_openpose_model, process_with_openpose
from sort.sort import Sort  # SORT 추가
from st_gcn.stgcn_runner import STGCNRunner  # ST-GCN 실행 모듈

# 경로 설정
YOLO_WEIGHTS = 'yolov9/weights/gelan-c-det.pt'
OPENPOSE_MODEL_PATH = 'pytorch_openpose/model/body_pose_model.pth'
ST_GCN_MODEL_PATH = 'st_gcn/model/st_gcn.ntu-xview.pt'

# ST-GCN에 입력하기 위한 데이터 전처리 함수
def preprocess_skeleton_data(skeleton_data, num_frames=300, num_joints=25, num_channels=3, num_persons=1):
    """
    스켈레톤 데이터를 ST-GCN 모델 입력 형식으로 전처리합니다.
    :param skeleton_data: OpenPose에서 얻은 스켈레톤 데이터
    :param num_frames: ST-GCN이 요구하는 프레임 수
    :param num_joints: 스켈레톤 관절 수
    :param num_channels: x, y, confidence
    :param num_persons: 추적된 사람 수
    :return: (N, C, T, V, M) 형태의 텐서
    """
    tensor = np.zeros((1, num_channels, num_frames, num_joints, num_persons))
    for frame_idx, skeleton in enumerate(skeleton_data[:num_frames]):
        for person_idx, joints in enumerate(skeleton[:num_persons]):
            for joint_idx, joint in enumerate(joints[:num_joints]):
                tensor[0, 0, frame_idx, joint_idx, person_idx] = joint[0]  # x
                tensor[0, 1, frame_idx, joint_idx, person_idx] = joint[1]  # y
                tensor[0, 2, frame_idx, joint_idx, person_idx] = joint[2]  # confidence
    return tensor

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

    # ST-GCN 모델 로드
    stgcn_runner = STGCNRunner(ST_GCN_MODEL_PATH)

    # 웹캠 초기화
    cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    print("Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        start_time = time.time()

        # YOLO 실행
        person_list, im0 = run_yolo(frame, YOLO_WEIGHTS)

        # 사람이 감지되지 않았을 때
        if len(person_list) == 0:
            print("No persons detected.")
            continue

        # SORT 입력 준비 (Bounding Box와 Confidence)
        detections = []
        skeleton_data = []

        for xyxy, crop_img in person_list:
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = 1.0  # YOLO에서 Confidence 추가 가능
            detections.append([x1, y1, x2, y2, confidence])

            # OpenPose 처리
            if y2 > y1 and x2 > x1:
                crop_img = im0[y1:y2, x1:x2].copy()

                # 이미지 유효성 검사
                if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                    print(f"Invalid crop_img detected: shape={crop_img.shape}")
                    continue

                try:
                    _, skeleton, _ = process_with_openpose(openpose_model, crop_img)
                    skeleton_data.append(skeleton)
                except Exception as e:
                    print(f"Error during OpenPose processing: {e}")

        # 감지된 객체가 없을 경우 빈 배열로 초기화
        if len(detections) == 0:
            detections = np.empty((0, 5))

        # SORT로 Tracking
        tracked_objects = tracker.update(np.array(detections))

        # Tracking 결과와 ST-GCN 처리
        for i, track in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = map(int, track)
            label = f"ID {obj_id}"
            color = (0, 255, 0)

            # Bounding Box 및 라벨 추가
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)

            # Skeleton 데이터와 행동 예측
            if i < len(skeleton_data):
                skeleton = skeleton_data[i]
                stgcn_input = preprocess_skeleton_data([skeleton])
                try:
                    action_label = stgcn_runner.predict_action(stgcn_input)
                    label += f" | {action_label}"
                except Exception as e:
                    print(f"Error during ST-GCN prediction: {e}")

            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS 계산
        fps = 1 / (time.time() - start_time)
        cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 결과 화면 출력
        cv2.imshow("YOLO + OpenPose + Tracking + ST-GCN (Webcam)", im0)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + OpenPose + Tracking + ST-GCN Webcam Demo")
    parser.add_argument(
        "-g", "--gpu_id", type=int, default=0, help="GPU ID to use (default: None for auto-selection)"
    )
    args = parser.parse_args()
    main(args)
