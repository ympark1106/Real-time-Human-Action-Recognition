import os
import cv2
import numpy as np
import threading
import queue
import argparse
from yolov9.yolo_runner import YoloRunner
from pytorch_openpose.openpose_runner import OpenPoseRunner
from sort.sort import Sort

# 경로 설정
YOLO_WEIGHTS = 'yolov9/weights/gelan-c-det.pt'
OPENPOSE_MODEL_PATH = 'pytorch_openpose/model/body_pose_model.pth'

# 작업 큐
frame_queue = queue.Queue(maxsize=5)
yolo_queue = queue.Queue(maxsize=5)
openpose_queue = queue.Queue(maxsize=5)
output_queue = queue.Queue(maxsize=5)

# SORT 객체
tracker = Sort()

# YOLO 처리 쓰레드
def yolo_thread():
    yolo_runner = YoloRunner(YOLO_WEIGHTS)
    while True:
        frame = frame_queue.get()
        if frame is None:
            yolo_queue.put(None)
            break
        person_list, im0 = yolo_runner.run(frame)
        yolo_queue.put((person_list, im0))

# OpenPose 처리 쓰레드
def openpose_thread():
    openpose_runner = OpenPoseRunner(OPENPOSE_MODEL_PATH)
    while True:
        yolo_result = yolo_queue.get()
        if yolo_result is None:
            openpose_queue.put(None)
            break
        person_list, im0 = yolo_result
        skeleton_data = []
        for xyxy, crop_img in person_list:
            x1, y1, x2, y2 = map(int, xyxy)
            if y2 > y1 and x2 > x1:
                pose_canvas, _, _ = openpose_runner.process(crop_img)
                skeleton_data.append((pose_canvas, (x1, y1, x2, y2)))
        openpose_queue.put((skeleton_data, im0))

# 결과 합성 및 트래킹 쓰레드
def tracking_thread(out):
    while True:
        openpose_result = openpose_queue.get()
        if openpose_result is None:
            output_queue.put(None)
            break
        skeleton_data, im0 = openpose_result

        detections = []
        for pose_canvas, (x1, y1, x2, y2) in skeleton_data:
            detections.append([x1, y1, x2, y2, 1.0])

        tracked_objects = tracker.update(np.array(detections))
        for track in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, track)
            label = f"ID {obj_id}"
            color = (0, 255, 0)
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for pose_canvas, (x1, y1, x2, y2) in skeleton_data:
            if im0[y1:y2, x1:x2].shape[:2] == pose_canvas.shape[:2]:
                im0[y1:y2, x1:x2] = cv2.addWeighted(im0[y1:y2, x1:x2], 0.5, pose_canvas, 0.5, 0)

        out.write(im0)
        output_queue.put(im0)

# 프레임 캡처 쓰레드
def capture_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)

# 결과 출력 쓰레드
def display_output():
    while True:
        im0 = output_queue.get()
        if im0 is None:
            break
        cv2.imshow("YOLO + OpenPose + Tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 메인 함수
def main(args):
    # GPU 설정
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU ID: {args.gpu_id}")
    else:
        print("Using default GPU (or CPU if no GPU available).")

    # 비디오 캡처 초기화
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

    # 쓰레드 생성
    threads = [
        threading.Thread(target=capture_frames, args=(cap,)),
        threading.Thread(target=yolo_thread),
        threading.Thread(target=openpose_thread),
        threading.Thread(target=tracking_thread, args=(out,)),
        threading.Thread(target=display_output)
    ]

    # 쓰레드 시작
    for thread in threads:
        thread.start()

    # 쓰레드 종료 대기
    for thread in threads:
        thread.join()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + OpenPose + Tracking Video Processor")
    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="GPU ID to use (default: None for auto-selection)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="output_with_skeleton.mp4", help="Path to save the output video")
    args = parser.parse_args()
    main(args)
