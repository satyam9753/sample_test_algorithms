import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/satyam/deep_learning/face_detection/Haar_Cascade/haar_cascade_frontal_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/satyam/deep_learning/face_detection/Haar_Cascade/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

#REFERENCE : https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    
    color = (255,0,0)  #(255,0,0) is for BLUE color

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness = 2) 

        dist = 6450 / w
        dist = '%.3f' % dist

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(dist), (x, y-10), font, 1, (0, 50, 250), 1, cv2.LINE_AA)

    cv2.imshow('img', img)

    if cv2.waitKey(20) & 0xff == 27 :
        break

cap.release()
