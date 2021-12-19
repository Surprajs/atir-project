import numpy as np
from tensorflow.keras.models import load_model
# import cv2
with np.load("atir_dataset.npz") as data:
    train_faces, train_labels = data["train_faces"], data["train_labels"]

model = load_model("facenet_keras.h5")
print(train_faces.shape)

def get_embedding(model, face):
    mean, std = np.mean(face), np.std(face)
    standard_face = (face-mean)/std
    samples = np.expand_dims(standard_face, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

new_train_faces = list()
for face in train_faces:
    embedding = get_embedding(model,face)
    new_train_faces.append(embedding)
    # print(face)
new_train_faces = np.asarray(new_train_faces)


embed_michal = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "piechowski"], axis=0)
embed_milosz = np.mean([face for face,label in zip(new_train_faces, train_labels) if label == "werner"], axis=0)


np.savez_compressed("atir_embeddings.npz", new_train_faces=new_train_faces, train_labels=train_labels)
np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal, embed_milosz=embed_milosz)

