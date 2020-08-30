# -*- coding: utf-8 -*-
# test-detection.py

import numpy as np
import cv2
import time
import os
import mss

def gaussian(image):
        kernel_size = 5
        return cv2.GaussianBlur(image, (kernel_size,kernel_size),0)
	
def canny(image, low_threshold, high_threshold):
        return cv2.Canny(image, low_threshold, high_threshold)

def nothing(x):
        pass

def mask(image):
        #create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(image)
        #define a four sided polygon to mask
        imshape = image.shape
        if len(imshape) > 2:
        	channel_count = imshape[2]
        	ignore_mask_color = (255,)*channel_count
        else:
        	ignore_mask_color=255

        vertices = np.array([[(320,1080),(400,500),(1200,500),(1280,1080)]],dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

def hough(image):
        rho=2
        theta=np.pi/180
        threshold=19
        min_line_length=24
        max_line_gap = 20
        return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),min_line_length,max_line_gap)

def draw_lanes(lines, image):
        lane_image = np.copy(image)*0 #creating a blank to draw lines on
        lane_color = (0,0,255)
        lane_thickness = 10
        
        left_slopes = []
        left_intercepts = []
        right_slopes = []
        right_intercepts = []
        
        y_max = image.shape[0]
        y_min = lane_image.shape[0]
        try:
           for line in lines:
               for x1,y1,x2,y2 in line:
                   m, b = np.polyfit((x1, x2), (y1, y2), 1)
                   y_min = min(y_min, y1, y2)
                   if(m > 0):
                       left_slopes.append(m)
                       left_intercepts.append(b)
                   if(m < 0):
                       right_slopes.append(m)
                       right_intercepts.append(b)
        except TypeError as e:
            print('TypeError')
            return image
        finally:
            print('others1')
        
        if len(left_slopes) > 0:
            # Draw the left lane
            left_slope = np.median(left_slopes)
            left_intercept = np.median(left_intercepts)
            left_x_min = int((y_min-left_intercept)/left_slope)
            left_x_max = int((y_max-left_intercept)/left_slope)
            try:
                cv2.line(lane_image, (left_x_min, y_min), (left_x_max, y_max), lane_color, lane_thickness)
            except OverflowError as e:
                print('OverflowError')
                return image
            finally:
                print('others')
            
        if len(right_slopes) > 0:
            # Draw the right lane
            right_slope = np.median(right_slopes)
            right_intercept = np.median(right_intercepts)
            right_x_min = int((y_min-right_intercept)/right_slope)
            right_x_max = int((y_max-right_intercept)/right_slope)
            try:
                cv2.line(lane_image, (right_x_min, y_min), (right_x_max, y_max), lane_color, lane_thickness)
            except OverflowError as e:
                print('OverflowError')
                return image
            finally:
                print('others')
        try:		
            cv2.line(lane_image, (400, 600), (400, 300), lane_color, lane_thickness)
        except OverflowError as e:
            print('OverflowError')
            return image
        finally:
            print('others')
        return cv2.addWeighted(image, 0.8, lane_image, 1, 0)

cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
cv2.createTrackbar("High", "detection",0,255, nothing)
cv2.createTrackbar("Low", "detection",0,255, nothing)

if __name__ == '__main__':
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    with mss.mss(display=':0.0') as sct:
        region={'top': 0, 'left': 0, 'width': 1960, 'height': 1080}
        while(True):
            last_time = time.time()
            screen = np.array(sct.grab(region))
            hul=cv2.getTrackbarPos("High", "detection")
            huh=cv2.getTrackbarPos("Low", "detection")

            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            filtered = gaussian(gray)
            #edges = canny(filtered, hul, huh)
            #cv2.imshow('canny',edges)

            #masked = mask(edges)
            #cv2.imshow('masked',masked) 
            #lines = hough(masked)
            #lineImage = draw_lanes(lines, screen)

            print('Frame took {} seconds'.format(time.time()-last_time))
            cv2.imshow("detection", filtered)
            #cv2.imshow("detection", lineImage)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindow()
                break
