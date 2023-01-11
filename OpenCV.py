import cv2 as cv
import numpy as np

# img = cv.imread('Photos/Cat.jpg')
# cv.imshow('Orignal', img)

# # # #image reading

# # # # img = cv.imread('Photos/Cat.jpg')
# # # # cv.imshow('Cat', img)
# # # # cv.waitKey(0)

# # # #---->Video Reading

# # # # capture = cv.VideoCapture('Videos/Dog.mp4')

# # # # while True:
# # # #     isTrue, frame = capture.read()
# # # #     cv.imshow("Video of Dog", frame)

# # # #     if cv.waitKey(20) & 0xFF == ord('d'):
# # # #         break

# # # # capture.release()
# # # # cv.destroyAllWindows()


# # # #resize video or image 

# # # def rescaleFrame(frame, scale = 0.75):
# # #     width = int(frame.shape[1] * scale)
# # #     height = int(frame.shape[0] * scale)

# # #     dimensions = (width,height)

# # #     return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# # # capture = cv.VideoCapture('Videos/Dog.mp4')

# # # #----->video Resize

# # # while True:
# # #     isTrue, frame = capture.read()

# # #     resized_video = rescaleFrame(frame, scale=0.2)
# # #     cv.imshow("Video", frame)
# # #     cv.imshow("Resized Video", resized_video)  

# # #     if cv.waitKey(20) & 0xFF == ord("D"):
# # #         break

# # # capture.release()
# # # cv.destroyAllWindows()

# # # #---->image resize

# # # # resized_img = rescaleFrame(img)
# # # # cv.imshow('Resized Image', resized_img)
# # # # cv.waitKey(0)

# # #-------> DRAW ON IMAGE A RECTANGLE

# # import cv2 as cv
# # import numpy as np

# # blank = np.zeros((500,500,3), dtype='uint8')
# # # blank[:] = 0,255,0
# # cv.rectangle(blank, (0,0), (500,250), (0,255,0), thickness=cv.FILLED)
# # # cv.imshow("Blank", blank)
# # # cv.waitKey(0)

# # # --------> DRAW A CIRCLE

# # cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 45, (0,0,255), thickness=-1)
# # # cv.imshow('Circle', blank)
# # # cv.waitKey(0)

# # # --------> DRAW A LINE

# # cv.line(blank, (0,500), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=2)
# # # cv.imshow('Blank', blank)
# # # cv.waitKey(0)

# # # --------> HOW TO PUT TEXT

# # cv.putText(blank, "HELLO",(0,280), cv.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)
# # cv.imshow('TEXT', blank)
# # cv.waitKey(0)

# # # --------> CONVERT IMAGE 2 GRAYSCALE

# # img = cv.imread('Photos/Cat.jpg')
# # cv.imshow('Orignal', img)

# # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # cv.imshow('GRAY', gray)

# # # --------> HOW TO BLUR IMAGE

# blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
# # cv.putText(blur, 'Blur Image', (0,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0),2)
# # cv.imshow("BLUR", blur)

# # # --------> EDGE CASCADE

# canny = cv.Canny(img, 130,175)
# # cv.imshow("Canyy", canny)

# # # #---->Crop Image

# cropped = img[100:600, 400:500]
# # cv.imshow("cropped", cropped)

# # # #---->Resize Image

# resized = cv.resize(img, (500,500), interpolation= cv.INTER_CUBIC)
# # cv.imshow("RESIZED", resized)

# # # #---->Translation of image (Shift or move along with X-axis or y-axis)

# def translate(img, x,y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dimension = (img.shape[1], img.shape[0])

#     return cv.warpAffine(img, transMat, dimension)

# img = cv.imread('Photos/park.jpg')
# translated = translate(img, 100, 100)
# cv.imshow("Translated", translated)

# # # #---->Flipping image

img = cv.imread('Photos/Cat.jpg')

# flipped = cv.flip(img, 0)
# cv.imshow("flipped", flipped)

# # # #---->Rotation of image

# def rotation(img, angle, rotPoint=None):
#     (width, height) = img.shape[:2]

#     if rotPoint is None:
#         rotPoint = (width//2, height//2)

#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
#     dimension = (width, height)

#     return cv.warpAffine(img, rotMat, dimension)

# rotated = rotation(img, 45)
# cv.imshow("Rotated", rotated)




















cv.waitKey(0)
