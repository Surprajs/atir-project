
from mtcnn import MTCNN
import cv2
from glob import glob
import numpy as np



def extract_face(filename, detector, size=(160,160)):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # for pyplot
    results = detector.detect_faces(img)
    x1,y1,width,height = results[0]['box']
    face = img[y1:y1+height,x1:x1+width]
    print(filename)
    return cv2.resize(face, size)

def load_faces(directory, detector, extension="jpg"):
    faces = list()
    images = glob(f"{directory}/*")
    for image in images:
        face = extract_face(image, detector)
        faces.append(face)
    return faces

def load_dataset(directory):
    detector = MTCNN()
    dirs = glob(f"{directory}/*/")
    print(dirs)
    dataset = list()
    labels = list()

    for dir in dirs:
        faces = load_faces(dir, detector)
        label = [dir.split("/")[-2]]*len(faces)
        labels.extend(label)
        dataset.extend(faces)

    return np.asarray(dataset), np.asarray(labels)



train_faces, train_labels = load_dataset("data")

np.savez_compressed("atir_dataset.npz", train_faces=train_faces, train_labels=train_labels)