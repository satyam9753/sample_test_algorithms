import cv2
import os
import numpy as np 

face_cascade = cv2.CascadeClassifier('/home/satyam/deep_learning/face_detection/Haar_Cascade/haar_cascade_frontal_default.xml')

#img = cv2.imread('')
cap = cv2.VideoCapture(0)
while True:

	_, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 4)

	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=3)

	cv2.imshow('img', img)
	
	if cv2.waitKey(20)& 0xFF == 27 :
		break

cap.release()
