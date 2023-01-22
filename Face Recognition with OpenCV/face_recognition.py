import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Aklif', 'Elton John', 'Haleem Sultan', 'Imran Khan', 'Jerry Seinfield']

# features = np.load("features.npy", allow_pickle=True)
# labels = np.load("labels.npy",  allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Sample image here
img = cv.imread("D:\\Python\\OpenCV\\Faces\\Validation\\Imran.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

# face Detect Here
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    face_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(face_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (10,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv.imshow("Recognized face", img)
cv.waitKey(0)