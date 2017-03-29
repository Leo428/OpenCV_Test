'''
Created on Mar 25, 2017
@author: zheyuan
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
font = cv2.FONT_ITALIC
while True:

    # Take each frame
    #frame = cv2.imread('1ftH2ftD2Angle0Brightness.jpg')
    _, frame = cap.read()
    cv2.line(frame, (320, int(0.25*480)),
             (320, int(0.75*480)), (0, 0, 255), 3)
    cv2.line(frame, (int(0.25*640), 240),
             (int(0.75*640), 240), (0, 0, 255), 3)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    # 139 155 128
    # 151  60 248
    #lower_blue = np.array([130,145,100])#np.array([170, 0, 0]) 
    lower_blue = np.array([78 ,43, 46])# peg
    #upper_blue = np.array([145,190,140])#np.array([200, 255, 255]) 
    upper_blue = np.array([99, 255, 255]) # peg
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #Addition:
    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 1:
        area_List = [0]
        centers_x = [0]
        centers_y = [0]
        
        for i in range(len(cnts)):
            c = cnts[i]
            area = cv2.contourArea(c)
            #area_List.append(area)
            if 250 < area < 30000:
                hull = cv2.convexHull(c)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                goal = cv2.approxPolyDP(hull, epsilon, True)
                
                cv2.drawContours(res, [goal], 0, (255, 0, 0), 5)

                M = cv2.moments(goal)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    center = (cx, cy)

                    cv2.circle(res, center, 5, (255, 0, 0), -1)
                    area_List.append(area)
                    centers_x.append(cx)
                    centers_y.append(cy)
#                 else:
#                     area_List.pop(i+1) #I will explain later
        
        if len(centers_x) == 3 and len(centers_y) == 3:
            target_x = (centers_x[1] + centers_x[2])/2
            target_y = (centers_y[1] + centers_y[2])/2
            target = (int(target_x), int(target_y))
            cv2.circle(res, target, 5, (0, 255, 0), -1)
            print("x " + str(target_x))
            print("y " + str(target_y))

            error = target_x - 340
            angle_to_turn = error * (59.02039664/640)
            print(angle_to_turn)
            aligned = 1 > angle_to_turn > -1
            #Display
            res = cv2.putText(res,("angle to turn: " + str(angle_to_turn)),(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)
            
        if len(centers_x) > 3 and len(centers_y) > 3:
            areaMax1 = (area_List.index(max(area_List)))
            areaMax2 = (area_List.index(max(n for n in area_List if n!=area_List[areaMax1])))
            cX1 = (centers_x[areaMax1])
            cY1 = (centers_y[areaMax1])
            cX2 = (centers_x[areaMax2])
            cY2 = (centers_y[areaMax2])
            target_x = (cX1 + cX2)/2
            target_y = (cY1 + cY2)/2
            target = (int(target_x), int(target_y))
            cv2.circle(res, target, 5, (0, 255, 0), -1)
            print("x " + str(target_x))
            print("y " + str(target_y))
 
            error = target_x - 340
            angle_to_turn = error * (59.02039664/640)
            print(angle_to_turn)
            aligned = 1 > angle_to_turn > -1
            #Display
            res = cv2.putText(res,("angle to turn: " + str(angle_to_turn)),(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)

    #Display image/window
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()