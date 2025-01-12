# http://localhost:5000

import os
import cv2
import time
import numpy as np
import sys
import threading
from flask import Flask, render_template, Response
from yolov9.yolo_runner import YoloRunner
from pytorch_openpose.openpose_runner import OpenPoseRunner
from sort.sort import Sort  # SORT 추가

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 경로 설정
YOLO_WEIGHTS = 'yolov9/weights/gelan-c-det.pt'
OPENPOSE_MODEL_PATH = 'pytorch_openpose/model/body_pose_model.pth'

# YOLO, OpenPose 초기화
yolo_runner = YoloRunner(YOLO_WEIGHTS)
openpose_runner = OpenPoseRunner(OPENPOSE_MODEL_PATH)
tracker = Sort()

# 웹캠 초기화
cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
if not cap.isOpened():
    raise RuntimeError("Error: Unable to open webcam.")

# 쓰레드 동기화 및 공유 변수
frame_lock = threading.Lock()
current_frame = None
output_frame = None


def process_frames():
    global current_frame, output_frame

    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        start_time = time.time()

        # YOLO 실행
        person_list, im0 = yolo_runner.run(frame)

        # 사람이 감지되지 않았을 때
        if len(person_list) == 0:
            with frame_lock:
                output_frame = frame  # 원본 프레임 그대로 저장
            continue

        # SORT 및 OpenPose 처리
        detections = []
        skeleton_data = []
        for xyxy, crop_img in person_list:
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = 1.0
            detections.append([x1, y1, x2, y2, confidence])

        if len(detections) > 0:
            tracked_objects = tracker.update(np.array(detections))
            for track in tracked_objects:
                x1, y1, x2, y2, obj_id = map(int, track)
                label = f"ID {obj_id}"
                color = (0, 255, 0)

                # Bounding Box
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # OpenPose 실행
                try:
                    crop_img = im0[y1:y2, x1:x2]
                    pose_canvas, _, _ = openpose_runner.process(crop_img)
                    if pose_canvas is not None:
                        pose_canvas = cv2.resize(pose_canvas, (x2 - x1, y2 - y1))
                        im0[y1:y2, x1:x2] = cv2.addWeighted(im0[y1:y2, x1:x2], 0.5, pose_canvas, 0.5, 0)
                except Exception as e:
                    print(f"OpenPose Error: {e}")

        # FPS 계산
        fps = 1 / (time.time() - start_time)
        cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 처리된 프레임 저장
        with frame_lock:
            output_frame = im0


def generate_frames():
    global output_frame

    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()

        # 프레임 인코딩 및 전송
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # 프레임 처리 쓰레드 시작
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()

    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port=5000, debug=True)

    # 웹캠 해제
    cap.release()
