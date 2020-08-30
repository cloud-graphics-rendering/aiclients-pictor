import sys
import os
import time
import cgrAPI
import cv2
import mss
import numpy as np
import keyboard_action
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from datetime import datetime
from TOD_Universal import TOD_Universal

def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]
    src = np.float32([[843, 518],    	# br
                      [1067, 519],    	# bl
                      [321, 900],   	# tl
                      [1530, 900]])  	# tr
    dst = np.float32([[843, 518],       # br
                      [1067, 519],      # bl
                      [843, 900],       # tl
                      [1067, 900]])     # tr
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def ProperBricksEdgePoint(box, brickLongEdgePoint):
    """
    Filter PROPER size rectangles and get the endPoint of longer edge.
    """
    dist1 = cgrAPI.pointsdist(box[0], box[1])
    dist2 = cgrAPI.pointsdist(box[1], box[2])
    if(dist1>dist2 and dist2>10 and dist1<400):
       brickLongEdgePoint.append([box[0],box[1]])
    elif(dist1<dist2 and dist1>10 and dist2 <400):
       brickLongEdgePoint.append([box[1],box[2]])
    else:
       pass
    return brickLongEdgePoint

def calculate_angle(endPoint):
    v0=np.array(endPoint[0])-np.array(endPoint[1])
    v1=np.array([0,1])-np.array([0,0])
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def produceLSTMInput(screen, lstmInputVec):
    last_angle = lstmInputVec[1]
    img_birdeye, M, Minv = birdeye(screen, verbose=True)
    huh=95
    hul=64
    gray = cv2.cvtColor(img_birdeye, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,hul,huh,1)
    _,contours,h = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    brickLongEdgePoint=[]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_birdeye,[box],0,(0,0,255),2)
            brickLongEdgePoint = ProperBricksEdgePoint(box, brickLongEdgePoint)
        else:
            pass
    if(len(brickLongEdgePoint) != 0):# refer the last one.
        angle = 0
        for points in brickLongEdgePoint:
            angle += calculate_angle(points)
        angle = angle/len(brickLongEdgePoint)
        lstmInputVec = [last_angle, angle]
    return lstmInputVec, img_birdeye

if __name__ == '__main__':
    count 		 = 0
    supertuxkart_restart = [985,985,1030,1030]
    lstmInputVec = [0,0]
    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1,2,3,512,1,2)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath 		 	= cgrAPI.LSTMInit("supertuxkart", AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= cgrAPI.CNNInit("supertuxkart", AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("supertuxkart", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode, 3)

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")

            with tf.Session(graph=cnnDetection_graph) as cnnSession:
                start_time 	= time.time()
                cur_time 	= time.time()
                last_cur_time   = cur_time
                while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN == 0)):
                    count	+= 1
                    print(count)
                    if(count >= 100):
                        keyboard_action.rescue()
                        keyboard_action.mouse_click(supertuxkart_restart)
                        count 	= 0

                    gamescreen 	 = cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start 	 = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    lstmInputVec, lstm_outImag = produceLSTMInput(gamescreen, lstmInputVec)
                    cv_end 	 = time.time()

                    if(lstm_based_Bots == 1):
                        lstm_start 	  = time.time()
                        lstmRealInput	  = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                        onehot_pred 	  = lstmSession.run(lstmPred, feed_dict={lstmX: lstmRealInput})
                        onehot_pred_index = np.argmax(onehot_pred)
                        lstm_end       	  = time.time()
                        if onehot_pred_index == 0: 
                            keyboard_action.turn_left()
                        elif(onehot_pred_index == 2):
                            keyboard_action.turn_right()
                        else:
                            keyboard_action.go_up()
                    else:
                        if(lstmInputVec[1] > 15):
                            keyboard_action.turn_left1()
                        elif(lstmInputVec[1] < -15):
                            keyboard_action.turn_right1()
                        else:
                            keyboard_action.go_up1()

                    cur_time = time.time()   
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  

