import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Photos//cats.jpg')
cv.imshow("Cats", img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow("Blank", blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)


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

# blank2 = np.zeros(img.shape[:2], dtype="uint8")

# circle = cv.circle(blank2.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1 )
# rectangel = cv.rectangle(blank2.copy(), (img.shape[1]//2+150, img.shape[0]//2-100), (img.shape[1]//2, img.shape[0]//2), 255, -1)

# mask = cv.bitwise_or(circle, rectangel)

# masked_image = cv.bitwise_and(img, img, mask=mask)
# cv.imshow("Mask", masked_image)

# # # # --------> Gray Histogram in OpenCV

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# circle = cv.circle(blank.copy(), (blank.shape[1]//2, blank.shape[0]//2), 100, 255, -1)

# mask = cv.bitwise_and(gray, gray, mask=circle)
# cv.imshow("Masked image", mask)

# gray_hist = cv.calcHist([gray], [0], mask, [256], [0,225])

# plt.figure()
# plt.title("Gray scale histogram")
# plt.xlabel("Bins")
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,225])
# plt.show()

# # # # --------> Color Histogram in OpenCV

# plt.figure()
# plt.title("Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")
# colors = ('b', 'g', 'r')

# for i, item in enumerate(colors):
#     color_hist = cv.calcHist([img], [i], None, [256], [0, 225])
#     plt.plot(color_hist, color=item)
#     plt.xlim([0, 225])

# plt.show()

# # # # --------> Simple Thresh holding in OpenCV

# threshold, thresh = cv.threshold(gray, 150, 225, cv.THRESH_BINARY)
# cv.imshow("Thresholded Image", thresh)

# threshold, thresh_inv = cv.threshold(gray, 150, 225, cv.THRESH_BINARY_INV)
# # cv.imshow("Thresholded Image Inverse", thresh_inv)
# print({threshold})

# # # # --------> Adaptive Threshholding in OpenCV

# adaptive_threshold = cv.adaptiveThreshold(gray, 150, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
# cv.imshow("Adaptive threshold image", adaptive_threshold)

# # # # --------> Laplacian

# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow("Laplacian", lap)

# # # # --------> Sobel 

# sobelX = cv.Sobel(gray, cv.CV_64F, 0, 1)
# sobelY = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobeled = cv.bitwise_or(sobelX, sobelY)

# cv.imshow("Sobeld", sobeled)

cv.waitKey(0)