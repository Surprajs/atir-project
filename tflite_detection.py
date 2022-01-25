import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from pycoral.adapters import common, detect, classify
from time import time
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PTUController import PTUController

def check_depth(x1,x2,y1,y2,depth_frame, color_img):
    try:
        print("test")
        x12 = (x1+x2)//2
        y12 = (y1+y2)//2
        w = abs(x2-x1)
        h = abs(y2-y1)
        nx1 = x1+w//4
        ny1 = y1+h//4
        nx2 = x2-w//4
        ny2 = y2-h//4
        depth_arr = np.array([[depth_frame.get_distance(x,y) for x in range (nx1,nx2+1)] for y in range(ny1,ny2+1)])
        avg = np.mean(depth_arr[depth_arr!=0])
        flat_factor = abs(np.sum((depth_arr[depth_arr!=0]-avg)*abs(depth_arr[depth_arr!=0]-avg))/len(depth_arr[depth_arr!=0]))
        return flat_factor > 1e-5
    except Exception as e:
        print(e)
        return False

def draw_boxes(color_img,depth_frame, objects, depth):
    if objects:
        for obj in objects:
            x1,y1,x2,y2 = obj.bbox
            if depth:
                if check_depth(x1,x2,y1,y2,depth_frame, color_img):
                    cv2.rectangle(color_img, (x1,y1),(x2,y2),(0,0,255),2)
            else:
                cv2.rectangle(color_img, (x1,y1),(x2,y2),(0,0,255),2)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--depth',type=bool,default=False,help='Use depth when detecting faces')
args = parser.parse_args()
            


#create the model
interpreter_detect = make_interpreter("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
interpreter_detect.allocate_tensors()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

WIDTH = 1280
HEIGHT = 720

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)



# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

FRAMES = 100
try:
    start = time()
    counter = 0
    while True:
        counter += 1
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        #aligned_frames = frames

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
        #if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        _, scale = common.set_resized_input(interpreter_detect, (WIDTH,HEIGHT), lambda size: cv2.resize(color_image, size))
        interpreter_detect.invoke()
        objects = detect.get_objects(interpreter_detect, 0.5, scale)
        
        draw_boxes(color_image,depth_frame, objects, args.depth)

        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        if counter == FRAMES:
            total_time = time() - start
            print(FRAMES/total_time)
            #break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
