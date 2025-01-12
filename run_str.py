# http://localhost:5000

import os
import cv2
import threading
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Error: Unable to open webcam.")

frame_lock = threading.Lock()
current_frame = None

def capture_frames():
    global current_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        with frame_lock:
            current_frame = frame.copy()
    cap.release()
    print("Webcam released.")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global current_frame

    while True:
        with frame_lock:
            if current_frame is None:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            frame = current_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
