import cv2 as cv
import numpy as np

img = cv.imread('Photos//cats.jpg')
cv.imshow("Cats", img)

blank = np.zeros((400,400), dtype='uint8')
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

# b, g, r = cv.split(img)

# blue = cv.merge([b, blank,blank])
# cv.imshow("BLue", blue)

# green = cv.merge([blank, g,blank])
# cv.imshow("Green", green)

# red = cv.merge([blank,blank,r])
# cv.imshow("Red", red)

# merged_image = cv.merge([b,g,r])
# cv.imshow("Merged Image", merged_image)

# # # # --------> How to blur images 

# average = cv.blur(img, (4, 4))
# # cv.imshow("Average", average)

# gaussian = cv.GaussianBlur(img, (3,3), 0)
# # cv.imshow("Gaussian", gaussian)

# median = cv.medianBlur(img, 1)
# # cv.imshow("Median blur", median)

# # # # --------> What are bitwise operations   

# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# circle = cv.circle(blank.copy(), (200,200), 200, 255, -1 )

# cv.imshow("Rectangle", rectangle)
# cv.imshow("Circle", circle)

# bitwise_and = cv.bitwise_and(rectangle, circle)
# cv.imshow("Bitwise AND", bitwise_and)

# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow("Bitwise OR", bitwise_or)

# bitwise_XOR = cv.bitwise_xor(rectangle, circle)
# cv.imshow("Bitwise XOR", bitwise_XOR)

# bitwise_NOT = cv.bitwise_not(rectangle)
# cv.imshow("Rectangle NOT", bitwise_NOT)

# # # # --------> What is Masking

blank2 = np.zeros(img.shape[:2], dtype="uint8")

mask = cv.circle(blank2, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1 )

masked_image = cv.bitwise_and(img, img, mask=mask)
cv.imshow("Mask", masked_image)







cv.waitKey(0)