import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from pycoral.adapters import common, detect, classify
from time import time

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PTUController import PTUController

def check_depth(x1,x2,y1,y2,depth_frame):
    try:
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
        return flat_factor > 1e-4
    except Exception as e:
        return False

def draw_boxes(color_img,depth_frame, objects, embeds, recognition_model):
    try:
        if objects:
            for obj in objects:
                x1,y1,x2,y2 = obj.bbox
                w = x2-x1; h = y2-y1
                face = color_img[y1+h//5:y2-h//5,x1+w//5:x2-w//5]
                sample = cv2.resize(face, (160,160))
                common.set_input(recognition_model, sample)
                recognition_model.invoke()
                embed = common.output_tensor(recognition_model, 0)[0]
                diff = [np.linalg.norm(embed-em, ord=2) for em in embeds]
                people = {0: "piechowski", 1: "hadrysiak"}
                if np.any(diff<3300):
                    idx = diff.index(min(diff))
                    cv2.rectangle(color_img, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(color_img, f"{people[idx]}", (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
                else:
                    cv2.rectangle(color_img, (x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(color_img, "unknown", (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)

    except Exception as e:
        print("error")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--depth',type=bool,default=False,help='Use depth when detecting faces')
args = parser.parse_args()
            
with np.load("embeddings.npz") as data:
    piechowski, hadrysiak = data["embed_piech"], data["embed_hadr"]
embeds = [piechowski, hadrysiak]
interpreter_detect = make_interpreter("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
interpreter_detect.allocate_tensors()

interpreter_recognize = make_interpreter("models/new_facenet_keras.tflite")
interpreter_recognize.allocate_tensors()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

WIDTH = 640
HEIGHT = 480

#config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)



# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        #aligned_frames =  frames

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
        #if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        _, scale = common.set_resized_input(interpreter_detect, (WIDTH,HEIGHT), lambda size: cv2.resize(color_image, size))
        interpreter_detect.invoke()
        objects = detect.get_objects(interpreter_detect, 0.5, scale)
        
        draw_boxes(color_image,0, objects, args.depth, embeds, interpreter_recognize, person)
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
