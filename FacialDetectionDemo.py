import cv2 as cv2
import numpy as np
import time
import pyrealsense2 as rs

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)


prev_frame_time = 0
new_frame_time = 0

try:
    while True:
        start = time.time()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)

        cv2.putText(color_image, fps, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)

        # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + w]

            cv2.putText(color_image, f'Person', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        cv2.imshow('Detected Face', color_image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
