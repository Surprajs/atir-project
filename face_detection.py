from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import cv2
from PIL import Image


interpreter = make_interpreter('models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
interpreter.allocate_tensors()

img = cv2.imread("img.png")
_, scale = common.set_resized_input(interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))

objs = detect.get_objects(interpreter,0.5,scale)

