import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from os import system
from time import time


pipeline = rs.pipeline()
config = rs.config()

WIDTH = 640
HEIGHT = 480

config.enable_stream(rs.stream.depth, WIDTH,HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH,HEIGHT, rs.format.bgr8, 30)

FRAMES = 100

# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
try:
    start = time()
    counter = 0
    while True:
        counter += 1
        frames = pipeline.wait_for_frames()
        #aligned_frames = align.process(frames)
        aligned_frames = frames
        #depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        #if not depth_frame or not color_frame:
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        #depth_image = np.asanyarray(depth_frame.get_data())

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        if counter == FRAMES:
            total_time = time() - start
            print(FRAMES/total_time)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()



