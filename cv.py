import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def image_detect(image_input):
    image = np.array(image_input, dtype=np.uint8)
    image=cv2.blur(image,(5,5))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([65, 70, 30])
    upper_green = np.array([90, 255, 255])
    image_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    image, contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(rect)
    detect=[]
    #0:x,1:y,2:w,3:h
    for rect in rects:
        if(rect[2]*rect[3]>=80 and rect[3]>12):
            cent_x=rect[0]+rect[2]/2
            cent_y=rect[1]+rect[3]/2
            detect.append([cent_x, cent_y, rect[2], rect[3]])
    return detect

def image_write(image,detect):
    strs=[]
    for p in detect:
        pos0=tuple([p[0]-p[2]/2,p[1]-p[3]/2])
        pos1=tuple([p[0]+p[2]/2,p[1]+p[3]/2])
        cv2.rectangle(image, pos0, pos1, (255, 0, 255), thickness=2)
        strs.append(str(p))
    y_pos=30
    for out_str in strs:
        cv2.putText(image, out_str, (0, y_pos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
        y_pos+=30
    return image

capture = cv2.VideoCapture(0)
    
if capture.isOpened() is False:
    print("IO Error")

else:
    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        detect=image_detect(image)
        image=image_write(image,detect)
        
        try:
            cv2.imshow("Capture", image)
        except:
            for i in sys.exc_info():
                print("ex error",i)
            print(sys.stderr)
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
        #if cv2.waitKey(33) >= 0:
            cv2.imwrite("image.png", image)
            break
    capture.release()
    cv2.destroyAllWindows()
