import cv2 as cv
import numpy as np

img = cv.imread('Photos//cats.jpg')
cv.imshow("Cats", img)

blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow("Blank", blank)

# # # --------> HOW TO FIND CONTOURS

# img = cv.imread('Photos//cats.jpg')
# cv.imshow("Cats", img)

# blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow("Blank", blank)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# # cany = cv.Canny(gray, 125, 175)
# # cv.imshow("Canny", cany)

# ret, thresh = cv.threshold(gray, 125,255, cv.THRESH_BINARY)
# cv.imshow("Threshold", thresh)

# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f"{len(contours)} contour(s) found!")

# cv.drawContours(blank, contours, -1, (0,255,0), 1)
# cv.imshow("Drawn Controus", blank)

# # # # --------> HOW TO SWITCH BETWEEN COLOR SPACES
# (1) - BGR to GrayScale
# (2) - BGR to HSV
# (3) - BGR to LAB
# (4) - BGR to RGB

# HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow("BGR ---> HSV", HSV)

# # # # --------> COLOR CHANELS

b, g, r = cv.split(img)

blue = cv.merge([b, blank,blank])
cv.imshow("BLue", blue)

green = cv.merge([blank, g,blank])
cv.imshow("Green", green)

red = cv.merge([blank,blank,r])
cv.imshow("Red", red)

merged_image = cv.merge([b,g,r])
cv.imshow("Merged Image", merged_image)








cv.waitKey(0)