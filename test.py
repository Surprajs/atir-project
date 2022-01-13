import cv2
from glob import glob
import numpy as np
from pycoral.adapters import common, detect, classify
from pycoral.utils.edgetpu import make_interpreter


def extract_face(img, detector, size=(160,160)):
    #img = cv2.imread(filename)
    h,w = img.shape[:-1]
    print(h,w)
    _, scale = common.set_resized_input(detector, (w,h), lambda size: cv2.resize(img, size))
    detector.invoke()
    objects = detect.get_objects(detector, 0.5, scale)
    if objects:
        x1,y1,x2,y2 = objects[0].bbox        
        #face = img[y1:y2,x1:x2]
        w = x2-x1; h = y2-y1
        face = img[y1+h//5:y2-h//5,x1+w//5:x2-w//5]
        cv2.imshow("face", face)
        cv2.waitKey(1000)
        #print(filename)
        return cv2.resize(face, size)
    else:
        return []

def get_embedding(face):
    model = make_interpreter("models/new_facenet_keras_edgetpu.tflite")
    model.allocate_tensors()
    #mean, std = np.mean(face), np.std(face)
    #standard_face = (face-mean)/std
    standard_face = face
    sample = np.expand_dims(standard_face, axis=0)
    common.set_input(model, sample)
    model.invoke()
    embed = common.output_tensor(model, 0)
    #print("here")
    embed = embed[0]
    print(embed[-5:])
    print(np.min(embed), np.max(embed))
    return embed



#train_faces, train_labels = load_dataset("dataset")


piechowski1 = cv2.imread("test/piech_fhd_1.png")
piechowski2 = cv2.imread("test/piech_fhd_2.png")
hadrysiak1 = cv2.imread("test/hadr_fhd_1.png")
hadrysiak2 = cv2.imread("test/hadr_fhd_2.png")



images = [piechowski1,piechowski2,hadrysiak1,hadrysiak2]
names2 = ["piechowski1","piechowski2","hadrysiak1","hadrysiak2"]

detector = make_interpreter("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
detector.allocate_tensors()
#model = make_interpreter("models/new_facenet_keras_edgetpu.tflite")
#model.allocate_tensors()
faces = [extract_face(image, detector) for image in images]
embeds = [get_embedding(face) for face in faces]

from itertools import combinations

dists = [(name, np.linalg.norm(pair[0]-pair[1], ord=2)) for pair, name in zip(combinations(embeds,2), combinations(names2,2))]

for dist in sorted(dists, key=lambda x: x[1]):
    print(dist)


