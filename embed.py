import cv2
from glob import glob
import numpy as np
from pycoral.adapters import common, detect, classify
from pycoral.utils.edgetpu import make_interpreter


def extract_face(filename, detector, size=(160,160)):
    img = cv2.imread(filename)
    _, scale = common.set_resized_input(detector, (1080,720), lambda size: cv2.resize(img, size))
    detector.invoke()
    objects = detect.get_objects(detector, 0.0, scale)
    if objects:
        x1,y1,x2,y2 = objects[0].bbox        
        #face = img[y1:y2,x1:x2]
        w = x2-x1; h = y2-y1
        face = img[y1+h//5:y2-h//5,x1+w//5:x2-w//5]
        print(filename)
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

def load_dataset(directory):
    detector = make_interpreter("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
    detector.allocate_tensors()
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
    print("here")
    print(embed)
    return embed



train_faces, train_labels = load_dataset("dataset2")



new_train_faces = list()
for face in train_faces:
    embedding = get_embedding(face)
    new_train_faces.append(embedding)
#print("before")
#print(new_train_faces)
#new_train_faces = np.array([embed for embed in new_train_faces])
#print("after")
#print(train_labels)
embed_michal = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "piechowski2"], axis=0)
print(embed_michal)
#embed_milosz = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "werner"], axis=0)
#print(embed_milosz)
#np.savez_compressed("atir_embeddings.npz", new_train_faces=new_train_faces[0], train_labels=train_labels)
np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal)
print("kurwa")
#np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal)#, embed_milosz=embed_milosz)

