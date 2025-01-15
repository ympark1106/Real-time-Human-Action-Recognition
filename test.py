import cv2
 
# Video4Linux2 백엔드(V4L2)를 사용하여 첫 번째 카메라 장치 열기
cap = cv2.VideoCapture(0)
 
if not cap.isOpened(cv2.CAP_V4L2):
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
 
    cv2.imshow('frame', frame)
 
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
