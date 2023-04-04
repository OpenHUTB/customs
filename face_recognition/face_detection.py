import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 调用摄像头输入图像
cap = cv2.VideoCapture(0)

frames = 0
start_time = time.time()

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # 画出检测结果
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    
    #计算FPS
    frames += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frames / elapsed_time
    cv2.putText(image, "FPS: {:.2f}".format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 水平翻转,调用前置摄像头时使用
    # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    
    cv2.imshow('MediaPipe Face Detection', image)

    # 按esc建退出检测
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
