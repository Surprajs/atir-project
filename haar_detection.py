import cv2 as cv2
import numpy as np
from time import time
import pyrealsense2 as rs

haar_cascade = cv2.CascadeClassifier('models/haar_face.xml')


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

fps_list = list()
try:
    while True:
        

        frames = pipeline.wait_for_frames()
        start = time()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if not color_frame:
            continue
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100,100))
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + w]

            cv2.putText(color_image, f'Person', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        cv2.imshow('Detected Face', color_image)
        fps_list.append(1/(time()-start))
        if len(fps_list) >= 100:
            with open("fps/fps_haar_det.txt", "w") as file:
                for fps in fps_list:
                    file.write(f"{fps}\n")
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
