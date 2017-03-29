'''
Created on Mar 28, 2017
@author: zheyuan
'''

import cv2
import numpy as np
import math
import os

#FRC Vision Target Calibration

if not os.path.isdir("/tmp/stream"):
    os.makedirs("/tmp/stream")

# Capture video from camera (0 for laptop webcam, 1 for USB camera)
cap = cv2.VideoCapture(1)

# Dimensions in use ()
FRAME_X = 640
FRAME_Y = 480
#FOV_ANGLE = 59.02039664
#DEGREES_PER_PIXEL = FOV_ANGLE / FRAME_X
FRAME_CX = int(FRAME_X/2)
FRAME_CY = int(FRAME_Y/2)

# Calibration box dimensions
CAL_AREA = 1600
CAL_SIZE = int(math.sqrt(CAL_AREA))
CAL_UP = FRAME_CY + (CAL_SIZE / 2)
CAL_LO = FRAME_CY - (CAL_SIZE / 2)
CAL_R = FRAME_CX - (CAL_SIZE / 2)
CAL_L = FRAME_CX + (CAL_SIZE / 2)
CAL_UL = (int(CAL_L), int(CAL_UP))
CAL_LR = (int(CAL_R), int(CAL_LO))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_X)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_Y)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
#cap.set(cv2.CAP_PROP_EXPOSURE, -8.0)

def main():
    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, CAL_UL, CAL_LR, (255, 0, 255), 3)
        roi = frame[int(CAL_LO):int(CAL_UP), int(CAL_R):int(CAL_L)]
        average_color_per_row = np.average(roi, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8([[average_color]])
        hsv = cv2.cvtColor(average_color, cv2.COLOR_BGR2HSV)

        print(np.array_str(hsv))
        cv2.imshow("Calibration", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("/tmp/stream/img.jpg", frame)
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
