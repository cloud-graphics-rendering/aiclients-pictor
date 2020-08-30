import sys
import os
import time
import cv2
import mss
import cgrAPI
import numpy as np
import keyboard_action
import keyboard
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from pymouse import PyMouse
from datetime import datetime
from TOD_Universal import TOD_Universal

if __name__ == '__main__':
    counter_n = 0
    m=PyMouse()
    redeclipse_center = [960, 564]
    offline_position = [1157, 1157, 523, 523]
    gametab_position = [593, 593, 132, 132]
    select_position  = [564, 564, 248, 248]
    disconnect_position  = [1184, 1184, 703, 703]
    output_keysList = []

    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1, 20, 5, 512, 0, 1)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath 	= cgrAPI.LSTMInit("redeclipse", AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 	= cgrAPI.CNNInit("redeclipse", AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("redeclipse", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode,5)

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph = cnnDetection_graph) as cnnSession:
                start_time 	= time.time()
                cur_time	= time.time()
                last_cur_time   = cur_time
                while (cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0):
                    lstm_start 	= 0
                    lstm_end 	= 0
                    counter_n = counter_n + 1
                    if counter_n >= 6000:
                        counter_n = 1
                        keyboard.press_and_release('esc')
                        time.sleep(1)
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end   = time.time()
                    obj_set = set(obj_classes)
                    m.click(redeclipse_center[0], redeclipse_center[1],2)
                    if 2 in obj_set:# disconnect
                        target_index = obj_classes.index(2)
                        target_position = obj_positions[target_index]
                        m.click(int((target_position[0]+target_position[1])/2), int((target_position[2]+target_position[3])/2))
                        m.click(redeclipse_center[0], redeclipse_center[1])
                    if 3 in obj_set:# offLinePlay
                        target_index = obj_classes.index(3)
                        target_position = obj_positions[target_index]
                        m.click(int((target_position[0]+target_position[1])/2), int((target_position[2]+target_position[3])/2))
                    if 5 in obj_set:# deathmatch
                        target_index = obj_classes.index(5)
                        target_position = obj_positions[target_index]
                        m.click(int((target_position[0]+target_position[1])/2), int((target_position[2]+target_position[3])/2))
                        m.click(redeclipse_center[0], redeclipse_center[1], 2)
                    elif 4 in obj_set:# favor
                        target_index = obj_classes.index(4)
                        target_position = obj_positions[target_index]
                        m.click(int((target_position[0]+target_position[1])/2), int((target_position[2]+target_position[3])/2))
                    elif 6 in obj_set:#enemy
                        map_index = obj_classes.index(6)
                        map_position = obj_positions[map_index]
                        if map_position[0] > redeclipse_center[0]:
                            keyboard_action.go_right()
                            m.move(redeclipse_center[0]+10,redeclipse_center[1])
                            m.click(redeclipse_center[0],redeclipse_center[1])
                        elif map_position[1] < redeclipse_center[0]:
                            keyboard_action.go_left()
                            m.move(redeclipse_center[0]-10,redeclipse_center[1])
                            m.click(redeclipse_center[0],redeclipse_center[1])
                        else:
                            m.click(redeclipse_center[0], redeclipse_center[1])
                        if map_position[2] > redeclipse_center[1]:
                            keyboard_action.go_backward()
                            m.move(redeclipse_center[0], redeclipse_center[0]+10)
                            m.click(redeclipse_center[0], redeclipse_center[1])
                        elif map_position[3] < redeclipse_center[1]:
                            keyboard_action.go_forward()
                            m.move(redeclipse_center[0], redeclipse_center[1]-10)
                            m.click(redeclipse_center[0], redeclipse_center[1])
                        else:
                            m.click(redeclipse_center[0], redeclipse_center[1])
                    else:#nothing is detected, try to find enemy
                        if lstm_based_Bots == 0 or len(output_keysList) < 20:
                            if counter_n % 300 == 0:
                                m.move(redeclipse_center[0] + 10, redeclipse_center[1])
                                output_keysList.insert(0,2)
                            elif counter_n % 200 == 0:
                                keyboard_action.go_forward()
                                m.click(redeclipse_center[0], redeclipse_center[1])
                                output_keysList.insert(0,3)
                            elif counter_n % 100 == 0:
                                keyboard_action.go_backward()
                                m.move(redeclipse_center[0]-10,redeclipse_center[1])
                                m.click(redeclipse_center[0], redeclipse_center[1])
                                output_keysList.insert(0,4)
                            else:
                                output_keysList.insert(0,1)
                                None
                            print("No enemy..Postion: %d,%d" %(m.position()))
                        else:
                            lstmInputVec    = output_keysList
                            lstm_start 	  = time.time()
                            lstmRealInput   = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                            onehot_pred     = lstmSession.run(lstmPred, feed_dict={lstmX: lstmRealInput})
                            onehot_pred_index = np.argmax(onehot_pred)
                            lstm_end 	  = time.time()
                            if onehot_pred_index == 0:#not move
                                m.click(redeclipse_center[0], redeclipse_center[1])
                                output_keysList.insert(0,0)
                                output_keysList.pop()
                                print("No enemy..not move")
                            elif onehot_pred_index == 1:#go left
                                keyboard_action.go_left()
                                output_keysList.insert(0,1)
                                output_keysList.pop()
                                print("No enemy..Left")
                            elif onehot_pred_index == 2:#go right
                                keyboard_action.go_right()
                                output_keysList.insert(0,2)
                                output_keysList.pop()
                                print("No enemy..Right")
                            elif onehot_pred_index == 3:#forward
                                keyboard_action.go_forward()
                                output_keysList.insert(0,3)
                                output_keysList.pop()
                                print("No enemy..Forward")
                            elif onehot_pred_index == 4:#backward
                                keyboard_action.go_backward()
                                output_keysList.insert(0,4)
                                output_keysList.pop()
                                print("No enemy..Backward")
                            else:
                                None
                    cur_time = time.time()   
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  
