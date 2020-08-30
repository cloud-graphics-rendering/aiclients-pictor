import sys
import os
import time
import cv2
import mss
import cgrAPI
import numpy as np
import keyboard
import pyautogui
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from pymouse import PyMouse
from datetime import datetime
from TOD_Universal import TOD_Universal

def centerOutofBox(region, point):
    if point[0] > region[1] or point[0] < region[0] or point[1] < region[2] or point[1] > region[3]:
        return True
    else:
        return False

if __name__ == '__main__':
    inmind_center = [940, 566]#960,564
    i_counter = 0
    last_drag = 0
    output_keysList = []
    lstmInputVec = [0,0]
    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1, 2, 3, 512, 1, 3)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode= cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath 		 	= cgrAPI.LSTMInit("inmindvr", AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= cgrAPI.CNNInit("inmindvr", AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("inmindvr", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode,3)

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph = cnnDetection_graph) as cnnSession:
                start_time = time.time()
                cur_time = time.time()
                last_cur_time   = cur_time
                while (cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0):
                    lstm_start = 0
                    lstm_end = 0
                    pyautogui.moveTo(inmind_center[0], inmind_center[1])
                    i_counter = i_counter+1
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end   = time.time()
                    obj_set = set(obj_classes)
                    if 1 in obj_set: #start
                        map_index = obj_classes.index(1)
                        map_position = obj_positions[map_index]
                        if centerOutofBox(map_position, inmind_center):
                            # drag 1 to center
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                    elif 3 in obj_set: #again
                        map_index = obj_classes.index(3)
                        map_position = obj_positions[map_index]
                        # drag 3 to center
                        if centerOutofBox(map_position, inmind_center):
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                            if i_counter > 1000:
                                pyautogui.moveTo(inmind_center[0], inmind_center[1])
                                i_counter = 0
                    elif 2 in obj_set: #web.com
                        map_index = obj_classes.index(2)
                        map_position = obj_positions[map_index]
                        # drag 3 to center
                        if centerOutofBox(map_position, inmind_center):
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]+80
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                            if i_counter > 1000:
                                pyautogui.moveTo(inmind_center[0], inmind_center[1])
                                i_counter = 0
                    else:
                        lower = np.array([171,0,21], dtype = "uint8")
                        upper = np.array([255,44,100], dtype = "uint8")
                        mask = cv2.inRange(gamescreen, lower, upper)
                        output = cv2.bitwise_and(gamescreen, gamescreen, mask = mask)
                        img1gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
                        retval1, img1bin = cv2.threshold(img1gray,50,255,0)
                        el     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        e_img  = cv2.erode(img1bin, el, iterations=1)
                        d_img  = cv2.dilate(e_img, el, iterations=20)
                        d_img,contours,hierarchy = cv2.findContours(d_img, 1, 2)
                        centerList = []
                        distList   = []
                        for contour in contours:
                            m_img  = cv2.moments(contour)
                            center = (int(m_img['m10']/m_img['m00']),int(m_img['m01']/m_img['m00']))
                            centerList.append(center)
                            distList.append(cgrAPI.pointsdist(center, inmind_center))
                        if lstm_based_Bots == 0:
                            if len(centerList) != 0:
                                center_closest = centerList[np.argmax(distList)]
                                pyautogui.dragRel(int(center_closest[0]-inmind_center[0]), int(center_closest[1]-inmind_center[1]),duration = 0.5)
                                time.sleep(1.5)
                            elif i_counter%2==0:
                                pyautogui.dragRel(0, -150,duration = 0.2)
                                pyautogui.dragRel(0, 150,duration = 0.2)
                            else:
                                pyautogui.dragRel(0, 150,duration = 0.2)
                                pyautogui.dragRel(0, -150,duration = 0.2)
                        else:
                            if len(centerList) != 0:
                                lstmInputVec    = [1,last_drag]
                            else:
                                lstmInputVec    = [0,last_drag]
                            lstmRealInput   = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                            lstm_start = time.time()
                            onehot_pred=lstmSession.run(lstmPred,feed_dict={lstmX: lstmRealInput})
                            lstm_end = time.time()
                            onehot_pred_index = np.argmax(onehot_pred)
                            if onehot_pred_index == 0:#target cell
                                center_closest = centerList[np.argmax(distList)]
                                pyautogui.dragRel(int(center_closest[0]-inmind_center[0]), int(center_closest[1]-inmind_center[1]),duration = 0.5)
                                time.sleep(1.5)
                            elif onehot_pred_index == 1:
                                pyautogui.dragRel(0, 150,duration = 0.2)
                                pyautogui.dragRel(0, -150,duration = 0.2)
                                last_drag = 0
                            elif onehot_pred_index == 2:
                                pyautogui.dragRel(0, -150,duration = 0.2)
                                pyautogui.dragRel(0, 150,duration = 0.2)
                                last_drag = 1
                            else:
                                None
                    cur_time = time.time()   
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  
