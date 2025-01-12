import cv2

for i in range(10):  # 최대 10개의 장치 확인
    cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
    if cap.isOpened():
        print(f"Webcam {i} is available.")
        cap.release()
    else:
        print(f"Webcam {i} is not available.")
