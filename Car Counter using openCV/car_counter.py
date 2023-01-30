import cv2 as cv
import numpy as np

# initialize variable

line_pos = 550
min_width = 80
min_height = 80
counter = 0
offset = 6

# Read the video

Capture = cv.VideoCapture(r"D:\Python\OpenCV\Car Counter using openCV\video.mp4")

# Initialize subtructor

algo = cv.bgsegm.createBackgroundSubtractorMOG()

# Creating function

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

detect = []

# Main Code

while True:
    ret, frame = Capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7,7), 5)

    # Applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv.dilate(img_sub, np.ones((12,12)))
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12))
    video_ROI = cv.morphologyEx(dilat, cv.MORPH_CLOSE, kernal)
    video_ROI = cv.morphologyEx(video_ROI, cv.MORPH_CLOSE, kernal)

    contours, h = cv.findContours(video_ROI, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw line and rectangle on video

    cv.line(frame, (25, line_pos), (1200, line_pos), (255,127,0), 3)
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv.boundingRect(c)
        validate_contour = (x>=min_width) and (x>=min_height)
        if not  validate_contour:
            continue

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # cv.putText(frame, "Vehicle:"+str(counter), (x, y-20), cv.FONT_HERSHEY_COMPLEX, 1, (124,0,255), 2)


        center = center_handle(x,y,w,h)
        detect.append(center)
        cv.circle(frame, center, 4, (0,0,255), -1)

        for (x,y) in detect:
            if y<(line_pos + offset) and y>(line_pos - offset):
                counter += 1
            cv.line(frame, (25, line_pos), (1200, line_pos), (0,127,255), 3)
            detect.remove((x,y))
            print(f"Number of vehicle: {counter}")

    cv.putText(frame, "No. of Vehicle:"+str(counter), (450,70), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)

    cv.imshow("Orignal Video", frame)

    if cv.waitKey(20) & 0xFF == ord(' '):
        break

cv.destroyAllWindows()
Capture.release()