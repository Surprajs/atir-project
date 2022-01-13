import cv2 as cv2
import numpy as np
from time import time
import pyrealsense2 as rs

haar_cascade = cv2.CascadeClassifier('models/haar_face.xml')


pipeline = rs.pipeline()
config = rs.config()

WIDTH = 1280
HEIGHT = 720

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
FRAMES = 100
try:
    start = time()
    counter = 0
    while True:
        counter += 1

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        #aligned_frames = frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
        #if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100,100))
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + w]

            cv2.putText(color_image, f'Person', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        cv2.imshow('Detected Face', color_image)
        if counter == FRAMES:
            total_time = time() - start
            print(FRAMES/total_time)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
