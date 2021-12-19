"""This script is responsible for identifying and analyzing multiple people's photos in order to train a specified
face recognition model.

The goal is to create a set of 3 separate files that will allow the FacialRecognitionDemo to perform a working face
detection and recognition in real time."""

import os
import cv2.cv2 as cv2
import numpy as np
import time
from datetime import timedelta

# Starts a timer that will display total time elapsed at the end of the program
start_time = time.monotonic()

# Finds catalogued people found in a specified directory and prints them
faces_dir = r'Faces'
people = list(set(os.listdir(r'Faces')) - {'desktop.ini', 'whatever.ini'})
print(f'Detected folders: {people}')

# Loading Haar Cascades for facial detection required to perform model training
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

# Creates empty arrays that will be filled with data patterns for each person's face (label)
features = []
labels = []


# Function that appends a list of features in order for the face recognition algorithm to work
def create_train():
    print(f'Total amount of people: {len(people)}')
    # Indexing every person detected in 'path' directory
    for person in people:
        path = os.path.join(faces_dir, person)
        label = people.index(person)

        photos = list(set(os.listdir(path)) - {'desktop.ini', 'whatever.ini'})
        total_photos = len(photos)

        print(f'Currently processing: {person}')
        print(f'Detected photos: {total_photos}')
        progress_bar(0, total_photos)
        iteration = 0

        # Reading every photo found in each person's individually labeled directory
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            if img_array is None:
                continue

            # Detecting the face in the photo
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Adding data to arrays within the detected face region
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

            # Updating the progress bar
            iteration += 1
            progress_bar(0 + iteration, total_photos)


def progress_bar(iteration, total, length=30):
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rProgress: |{bar}| {int(percent)}% Complete', end='')
    if iteration == total:
        print()
        print('Process done!')


# Running the training function and changing the arrays into a numpy array
create_train()
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# Removes the previously generated face recognition data
if 'face_trained.yml':
    os.remove('face_trained.yml')
    os.remove('features.npy')
    os.remove('labels.npy')

# Saving the face recognition data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print('--- Training done ---')
print(f"Time elapsed {timedelta(seconds=time.monotonic() - start_time)}")
