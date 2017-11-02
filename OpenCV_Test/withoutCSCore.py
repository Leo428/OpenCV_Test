'''
Created on Mar 28, 2017
@author: Zheyuan Hu
'''

import cv2
import numpy as np
from numpy import shape
from networktables import NetworkTables as nT 

'''
important things you want to learn:
please read and understand http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
'''

font = cv2.FONT_ITALIC
aligned = False
ip = '10.11.38.57'

def drawCrossHair(mFrame):
    # draw a crosshair
    cv2.line(mFrame, (320, int(0.25*480)),
             (320, int(0.75*480)), (0, 0, 255), 3)
    cv2.line(mFrame, (int(0.25*640), 240),
             (int(0.75*640), 240), (0, 0, 255), 3)

def sendAngle(angle):
    if nT.isConnected():
        print('Connection to robot: ' + str(nT.isConnected()))
        sd = nT.getTable('SmartDashboard')
        sd.putNumber('setAngle', angle)

def main():
    cap = cv2.VideoCapture(0)
    frame = np.zeros(shape = (480,640,3), dtype=np.uint8)
    while True:
        # Take each frame
        ret, frame = cap.read()
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define range of blue color in HSV
        #lower_blue = np.array([130,145,100])#np.array([170, 0, 0]) 
        lower_blue = np.array([45,43,46])# peg [78 ,43, 46]
        #upper_blue = np.array([145,190,140])#np.array([200, 255, 255]) 
        upper_blue = np.array([77,255,255]) # peg [99, 255, 255]
        
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame,frame, mask= mask)
    
        #Find the contours by Addition:
        _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

        # if there are possible targets (at least one)
        if len(cnts) > 1:
            # initialize the space to hold values 
            area_List = [0]
            centers_x = [0]
            centers_y = [0]
        
            for i in range(len(cnts)):
                c = cnts[i]
                area = cv2.contourArea(c)
                #area_List.append(area)
                if 1000 < area < 30000: # if the size is in this range
                    hull = cv2.convexHull(c)
                    epsilon = 0.02 * cv2.arcLength(hull, True)
                    goal = cv2.approxPolyDP(hull, epsilon, True)
                    cv2.drawContours(frame, [goal], 0, (255, 0, 0), 5) # draw a ractangle around the target 

                    # this is used to find the center of the target
                    M = cv2.moments(goal) # Centroid is given by the relations, Cx=M10/M00 and Cy=M01/M00.
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        center = (cx, cy)

                        #cv2.circle(res, center, 5, (255, 0, 0), -1)
                        cv2.circle(frame, center, 5, (255, 0, 0), -1)
                        area_List.append(area)
                        centers_x.append(cx)
                        centers_y.append(cy)
        
            if len(centers_x) == 3 and len(centers_y) == 3: # if there are two targets 
                target_x = (centers_x[1] + centers_x[2])/2
                target_y = (centers_y[1] + centers_y[2])/2
                target = (int(target_x), int(target_y))
                cv2.circle(frame, target, 5, (0, 255, 0), -1) # add a circle as center point 
                print("x " + str(target_x))
                print("y " + str(target_y))

                error = target_x - 320
                angle_to_turn = int(error * (60.0/800.0) + 0.5)
                sendAngle(angle_to_turn) # send angle to robot
                print('turn angle: ' + str(error) + ' ' + str(angle_to_turn))
                aligned = 1 > angle_to_turn > -1
                #Display
                frame = cv2.putText(frame,
                                  ("angle to turn: " 
                                   + str(angle_to_turn)),
                                  (10,400),
                                  font,
                                  1,
                                  (255,255,255),
                                  2,
                                  cv2.LINE_AA)
            
            if len(centers_x) > 3 and len(centers_y) > 3: # if there are more than 2 area detected
                # find two of the largest area as targets  
                areaMax1 = (area_List.index(max(area_List)))
                areaMax2 = (area_List.index(max(n for n in area_List if n!=area_List[areaMax1])))
                cX1 = (centers_x[areaMax1])
                cY1 = (centers_y[areaMax1])
                cX2 = (centers_x[areaMax2])
                cY2 = (centers_y[areaMax2])
                target_x = (cX1 + cX2)/2
                target_y = (cY1 + cY2)/2
                target = (int(target_x), int(target_y))
                cv2.circle(frame, target, 5, (0, 255, 0), -1) # add a circle as center point 
                print("x " + str(target_x))
                print("y " + str(target_y))
 
                error = target_x - 320
                angle_to_turn = int(error * (60.0/800.0) + 0.5)
                sendAngle(angle_to_turn) # send angle to robot
                print('turn angle: ' + str(error) + ' ' + str(angle_to_turn))
                aligned = 1 > angle_to_turn > -1
                #Display
                frame = cv2.putText(frame,
                                  ("angle to turn: " 
                                   + str(angle_to_turn)),
                                  (10,400),
                                  font,
                                  1,
                                  (255,255,255),
                                  2,
                                  cv2.LINE_AA)
                
        drawCrossHair(frame)
        cv2.imshow('Vision', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite("/tmp/stream/img.jpg", frame)
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    nT.initialize(server = ip)
    main()

    
    

