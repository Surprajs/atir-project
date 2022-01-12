import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from os import system
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name',type=str,default="nobody",help='name in the files')
args = parser.parse_args()

system(f"mkdir -p dataset/{args.name}")

pipeline = rs.pipeline()
config = rs.config()

WIDTH = 640
HEIGHT = 480

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)



# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        #cv2.imwrite("test.png", color_image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(20) & 0xFF == ord('s'):
            cv2.imwrite(f"dataset/{args.name}/{args.name}-{datetime.now().strftime('%H-%M-%S')}.png", color_image)
            print(datetime.now().strftime("%H:%M:%S"))

finally:

    # Stop streaming
    pipeline.stop()



