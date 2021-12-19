"""This script is responsible for facial detection and recognition in real time.

The goal is to detect a face in each frame and to identify what person said face belongs to."""

import cv2.cv2 as cv2
import numpy as np
import os
import time

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

people = list(set(os.listdir(r'Faces')) - {'desktop.ini', 'whatever.ini'})
print(f'Detected folders: {people}')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv2.VideoCapture(0)

scale_ratio = 4

prev_frame_time = 0
new_frame_time = 0

while capture.isOpened():
    isTrue, frame = capture.read()
    if not isTrue:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)

    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)

    # Detect the face in the frame
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, loss_of_confidence = face_recognizer.predict(faces_roi)
        confidence = 100 - loss_of_confidence
        if confidence < 0:
            confidence = 0
        print(f'\rLabel = {people[label]} with confidence = {confidence}', end='')

        cv2.putText(frame, f'{int(confidence)}%', (x - 40, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                    thickness=1)

        if confidence > 65:
            if str(people[label]) == "Milosz Werner":
                cv2.putText(frame, str(people[label]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),
                            thickness=1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
            if str(people[label]) == "Ryan Gosling":
                cv2.putText(frame, str(people[label]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0),
                            thickness=1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), thickness=1)

        else:
            cv2.putText(frame, f'Person', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

    cv2.imshow('Detected Face', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
