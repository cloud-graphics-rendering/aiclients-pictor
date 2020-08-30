import sys
import os
import time
import cv2
import mss
import numpy as np
import cgrAPI
import keyboard_action
import pyautogui
import keyboard
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from datetime import datetime
from PIL import Image
from TOD_Universal import TOD_Universal

def imhotepvrBotActions():
    initPos = [954, 954, 623, 623]
    patientPos = [928, 928, 482, 482]
    layerPos = [670, 670, 1003, 1003]
    liverMeshPos = [714, 714, 274, 274]
    XPos = [1244, 1244, 999, 999]
    time.sleep(6)
    keyboard_action.mouse_click(initPos)
    return

if __name__ == '__main__':
    dragPos = [681,681,679,679]
    initPos = [954, 954, 623, 623]
    patientPos = [928, 928, 482, 482]
    flag_count = 1
    last_flag_count = 0
    push_flag = 0
    onehot_pred_index = 0
    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1, 2, 4, 512, 1, 3)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath	= cgrAPI.LSTMInit("imhotepvr", AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 	= cgrAPI.CNNInit("imhotepvr", AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file						= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("imhotepvr", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode,3)
    imhotepvrBotActions()
    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")

            with tf.Session(graph=cnnDetection_graph) as cnnSession:
                start_time = time.time()
                cur_time = time.time()
                last_cur_time   = cur_time
                while (HUMAN_RUN == 0) and (cur_time - start_time <= RUNNING_TIME):
                    lstm_start	= 0
                    lstm_end	= 0
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end   = time.time()
                    obj_set = set(obj_classes)
                    if 1 in obj_set and push_flag == 0:
                        keyboard_action.mouse_click(initPos)
                    if 2 in obj_set and push_flag == 0:
                        keyboard_action.mouse_click(patientPos)
                        time.sleep(1)
                        keyboard_action.mouse_click(patientPos)
                        push_flag = 1
                    if push_flag == 1:
                        if 3 in obj_set:
                            push_flag = 2
                        else:
                            keyboard.press('up')
                            time.sleep(0.1)
                            keyboard.release('up')
                    if push_flag == 2:
                        if lstm_based_Bots == 1:
                            lstmInputVec    = [last_flag_count, flag_count]
                            last_flag_count = flag_count
                            lstmRealInput   = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                            lstm_start = time.time()
                            onehot_pred     = lstmSession.run(lstmPred, feed_dict={lstmX: lstmRealInput})
                            lstm_end = time.time()
                            onehot_pred_index = np.argmax(onehot_pred)
                            if onehot_pred_index == 0:
                                keyboard_action.moveTo(100, 679, duration = 0.1)
                                pyautogui.dragRel(1500, 0, duration = 1)
                                flag_count = 1
                            elif onehot_pred_index == 1:
                                keyboard_action.moveTo(1700, 679, duration = 0.1)
                                pyautogui.dragRel(-1500, 0, duration = 1)
                                flag_count = 2
                            elif onehot_pred_index == 2:
                                keyboard_action.moveTo(679, 100, duration = 0.1)
                                pyautogui.dragRel(0, 800, duration = 1)
                                flag_count = 0
                            elif onehot_pred_index == 3:
                                keyboard_action.moveTo(679, 900, duration = 0.1)
                                pyautogui.dragRel(0, -800, duration = 1)
                                flag_count  = 3
                            else:
                                None
                        else: 
                            if flag_count == 0:
                                keyboard_action.moveTo(100, 679, duration = 0.1)
                                pyautogui.dragRel(1500, 0, duration = 1)
                                flag_count += 1
                            elif flag_count == 1:
                                keyboard_action.moveTo(1700, 679, duration = 0.1)
                                pyautogui.dragRel(-1500, 0, duration = 1)
                                flag_count += 1
                            elif flag_count == 2:
                                keyboard_action.moveTo(679, 100, duration = 0.1)
                                pyautogui.dragRel(0, 800, duration = 1)
                                flag_count += 1
                            elif flag_count == 3:
                                keyboard_action.moveTo(679, 900, duration = 0.1)
                                pyautogui.dragRel(0, -800, duration = 1)
                                flag_count  = 0
                    cur_time   = time.time()
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  
