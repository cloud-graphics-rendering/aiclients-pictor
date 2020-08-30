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
    time.sleep(12)
    keyboard_action.mouse_click(patientPos)
    time.sleep(4)
    for i in range(10):
        keyboard.press('up');
        time.sleep(0.2);
        keyboard.release('up');
    keyboard_action.mouse_click(layerPos)
    time.sleep(1)
    keyboard_action.mouse_click(liverMeshPos)
    time.sleep(1)
    pyautogui.dragRel(-100, 0, duration = 1)
    time.sleep(2)
    keyboard_action.mouse_click(XPos)
    time.sleep(1)
    return

if __name__ == '__main__':
    file_name = '../training_data/raw-data/training_data' + str(int(time.time())) + '.npy'
    training_data = []
    output_keysList=[0,0]
    dragPos = [681,681,679,679]
    flag_count = 0
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    cgrAPI.commandInit("imhotepvr", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode)
    imhotepvrBotActions()

    start_time = time.time()
    cur_time = time.time()
    while (HUMAN_RUN == 0) and (cur_time - start_time <= RUNNING_TIME):
        # click start position
        pyautogui.moveTo(dragPos[0], dragPos[2], duration = 0.1)
        if flag_count == 0:
            flag_count += 1
            pyautogui.dragRel(300, 0, duration = 1)
            record = [[0,flag_count],[1,0,0,0]]
            training_data.append(record)
        elif flag_count == 1:
            flag_count += 1
            pyautogui.dragRel(-300, 0, duration = 1)
            record = [[1,flag_count],[0,1,0,0]]
            training_data.append(record)
        elif flag_count == 2:
            flag_count += 1
            pyautogui.dragRel(0, 300, duration = 1)
            record = [[2,flag_count],[0,0,1,0]]
            training_data.append(record)
        elif flag_count == 3:
            flag_count  = 0
            pyautogui.dragRel(0, -300, duration = 1)
            record = [[3,flag_count],[0,0,0,1]]
            training_data.append(record)
        else:
            None 
        if len(training_data) % 10 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
        
        cur_time   = time.time()

