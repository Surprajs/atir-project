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
        #print(filename)
        return cv2.resize(face, size)
    else:
        return []
def load_faces(directory, detector):
    faces = list()
    images = glob(f"{directory}/*")
    for image in images:
        face = extract_face(image, detector)
        if np.any(face):
            faces.append(face)
    return faces

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


milosz1 = cv2.imread("test/milosz1.jpg")
milosz2 = cv2.imread("test/milosz2.jpg")
michal1 = cv2.imread("test/michal1.jpg")
michal2 = cv2.imread("test/michal2.jpg")
#milosz1 = cv2.imread("test/test1.png")
#milosz2 = cv2.imread("test/test2.png")
#milosz1 = cv2.imread("test/camera1.png")
#milosz2 = cv2.imread("test/camera3.png")
michal1 = cv2.imread("test/test3.png")
michal2 = cv2.imread("test/test4.png")


images = [milosz1, milosz2, michal1, michal2]
names2 = ["milosz1", "milosz2", "michal1","michal2"]

detector = make_interpreter("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
detector.allocate_tensors()
#model = make_interpreter("models/new_facenet_keras_edgetpu.tflite")
#model.allocate_tensors()
faces = [extract_face(image, detector) for image in images]
embeds = [get_embedding(face) for face in faces]

from itertools import combinations

dists = [(name, np.linalg.norm(pair[0]-pair[1], ord=2)) for pair, name in zip(combinations(embeds,2), combinations(names2,2))]

for dist2 in sorted(dists, key=lambda x: x[1]):
    print(dist2)
#print(new_train_faces)
#new_train_faces = np.array([embed for embed in new_train_faces])
#print("after")
#print(train_labels)
#embed_michal = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "piechowski"], axis=0)
#print(embed_michal)
#embed_milosz = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "werner"], axis=0)
#print(embed_milosz)
#np.savez_compressed("atir_embeddings.npz", new_train_faces=new_train_faces[0], train_labels=train_labels)
#np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal)
#print("kurwa")
#np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal)#, embed_milosz=embed_milosz)

