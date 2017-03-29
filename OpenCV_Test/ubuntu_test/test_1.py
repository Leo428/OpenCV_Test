'''
Created on Mar 25, 2017
@author: zheyuan
'''
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)

img = np.zeros(shape = (480,640,3), dtype = np.uint8)
font = cv2.FONT_ITALIC
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read(img)
    frame = cv2.rectangle(img, (300, 100), (340, 400), (255, 255, 255), 5)
    #frame = cv2.putText(frame,str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
