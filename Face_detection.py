# import cv2 as cv

# img = cv.imread("Photos//me.jpg")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # cv.imshow("Gray person image", gray)

# haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# face_rect = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 9) 

# print(f"Number of faces : {len(face_rect)}")

# for (x,y,w,h) in face_rect:
#     cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 12)

# resize = cv.resize(img, [600,600], cv.INTER_AREA)

# cv.imshow("Detected face", resize)
# cv.waitKey(0)
