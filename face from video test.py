from enum import Flag
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 

face_cascade = cv2.CascadeClassifier('C:\\Users\\Matthew Toven\\Documents\\TUFTS\\Classes\\Probabilistic robotics\\haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while (True):

    ret, frame= vid.read()

    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()