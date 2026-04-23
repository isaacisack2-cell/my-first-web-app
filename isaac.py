import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print("press 'q' to escape")

while True:
    ret,frame = cap.read()
    if not ret:
        print("camera imeshindwa kufunguka kwa sasa")
        break

    flipped_vid = cv2.flip(frame,1)         #flipping the video
    blue_only = flipped_vid.copy()
    blue_only[:,:,1] = 0 #zima green
    blue_only[:,:,2] = 0    #zima red

    negative = 255 - flipped_vid        #inverting the color
    combined = np.hstack((flipped_vid, blue_only))      #using numpy kuunganisha picha mbili horizontally horizontal stack

    cv2.imshow("normal vs blue vision",combined)
    cv2.imshow('negative mode',negative)

    if cv2.waitKey(10000) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
