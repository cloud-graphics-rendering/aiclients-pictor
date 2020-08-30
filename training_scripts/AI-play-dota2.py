import sys
import os
import time
import cv2
import mss
import cgrAPI
import numpy as np
import keyboard_action
import keyboard
import keyboard_action
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from datetime import datetime
from PIL import Image
from TOD_Universal import TOD_Universal

def dotaBotActions():
    time.sleep(2)
    keyboard_action.mouse_click([1686, 1686, 1050, 1050])
    time.sleep(2)
    keyboard_action.mouse_click([1686, 1686, 1050, 1050])
    time.sleep(8)
    # choose hero luna
    keyboard_action.mouse_click([823, 823, 431, 431])
    time.sleep(4)
    keyboard_action.mouse_click([1481, 1481, 815, 815])
    time.sleep(8)
    # skip ahead
    keyboard_action.mouse_click([158, 158, 812, 812])
    time.sleep(6)
    # click start position
    keyboard_action.mouse_click([38, 38, 1055, 1055])
    keyboard_action.mouse_click([38, 38, 1055, 1055])
    for i in range(25):
        keyboard.press('up')
        keyboard.press('right')
        time.sleep(0.2)
        keyboard.release('up')
        keyboard.release('right')
        time.sleep(0.1)
    for i in range(12):
        keyboard.press('down')
        keyboard.press('left')
        time.sleep(0.2)
        keyboard.release('down')
        keyboard.release('left')
        time.sleep(0.1)
    keyboard_action.mouse_double_click([678, 678, 1015, 1015])

def retreatAction(battle_flag, retreat_flag):
    battle_flag = 0
    if(retreat_flag < 5):
        keyboard_action.moveTo(40, 1044, duration = 2)
        keyboard_action.mouse_click([40,40,1044,1044],2,0)
        retreat_flag += 1
    elif(retreat_flag==5):
        retreat_flag += 1
        keyboard_action.moveTo(startPos[0], startPos[-1], duration = 2)
        keyboard_action.mouse_click(startPos)
        for i in range(25):
            keyboard.press('up');
            keyboard.press('right');
            time.sleep(0.1);
            keyboard.release('up');
            keyboard.release('right');
            time.sleep(0.1);
        for i in range(12):
            keyboard.press('down');
            keyboard.press('left');
            time.sleep(0.1);
            keyboard.release('down');
            keyboard.release('left');
            time.sleep(0.1);
    else:
        keyboard_action.moveTo(player2Pos[0], player2Pos[-1], duration = 2)
        keyboard_action.mouse_click(player2Pos)
        time.sleep(1)
        keyboard_action.moveTo(commonPos[0], commonPos[-1], duration = 2)
        keyboard_action.mouse_double_click(commonPos)
        time.sleep(3)
        keyboard_action.moveTo(player3Pos[0], player3Pos[-1], duration = 2)
        keyboard_action.mouse_click(player3Pos)
        time.sleep(1)
        keyboard_action.moveTo(commonPos[0], commonPos[-1], duration = 2)
        keyboard_action.mouse_double_click(commonPos)
        time.sleep(3)
        keyboard_action.moveTo(player4Pos[0], player4Pos[-1], duration = 2)
        keyboard_action.mouse_click(player4Pos)
        time.sleep(1)
        keyboard_action.moveTo(commonPos[0], commonPos[-1], duration = 2)
        keyboard_action.mouse_double_click(commonPos)
        time.sleep(3)
        keyboard_action.moveTo(player5Pos[0], player5Pos[-1], duration = 2)
        keyboard_action.mouse_click(player5Pos)
        time.sleep(1)
        keyboard_action.moveTo(commonPos[0], commonPos[-1], duration = 2)
        keyboard_action.mouse_double_click(commonPos)
        time.sleep(3)
        keyboard_action.moveTo(player1Pos[0], player1Pos[-1], duration = 2)
        keyboard_action.mouse_click(player1Pos)
        time.sleep(1)
        keyboard_action.moveTo(commonPos[0], commonPos[-1], duration = 2)
        keyboard_action.mouse_double_click(commonPos)
    return battle_flag, retreat_flag

def battleAction(battle_flag, retreat_flag):
    retreat_flag = 0
    if(battle_flag < 5):
        keyboard_action.mouse_click([115,115,990,990],2,0)
        battle_flag += 1
    else:
        None
    return battle_flag, retreat_flag

if __name__ == '__main__':
    player1Pos 		= [574, 574, 44, 44]
    player2Pos 		= [634, 634, 44, 44]
    player3Pos 		= [695, 695, 45, 45]
    player4Pos 		= [758, 758, 43, 43]
    player5Pos 		= [821, 821, 42, 42]
    commonPos 		= [678, 678, 1015, 1015]
    startPos 		= [38, 38, 1055, 1055]
    position_vec = [0, 1080]
    x_dim 		= range(786,1096,1)
    last_life_value 	= 0
    retreat_flag 	= 0
    battle_flag 	= 0

    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1,4,3,512,1,2)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath			= cgrAPI.LSTMInit("dota2",AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= cgrAPI.CNNInit("dota2",AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("dota2", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode, 10)
    dotaBotActions()

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph=cnnDetection_graph) as cnnSession:
                start_time 	= time.time()
                cur_time 	= time.time()
                last_cur_time   = cur_time
                while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
                    lstm_start	= 0
                    lstm_end	= 0
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start    = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end     	= time.time()
                    obj_set = set(obj_classes)
                    if 1 in obj_set:
                        hero_index = obj_classes.index(1)
                        pos_hero = obj_positions[hero_index]
                        position_vec = [int((pos_hero[0]+pos_hero[1])/2), int((pos_hero[0]+pos_hero[1])/2)]
                    life_value = 0
                    for x in x_dim:
                        if gamescreen[1045,x,1] > 100:
                            life_value+=1
                    if lstm_based_Bots == 1:
                        lstmInputVec = position_vec + [last_life_value,life_value]
                        lstmRealInput   = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                        lstm_start    	= time.time()
                        onehot_pred     = lstmSession.run(lstmPred, feed_dict={lstmX: lstmRealInput})
                        lstm_end	= time.time()
                        onehot_pred_index = np.argmax(onehot_pred)
                        if onehot_pred_index == 2:#retreat
                            battle_flag, retreat_flag = retreatAction(battle_flag, retreat_flag)
                        elif onehot_pred_index == 0:#battle action
                            battle_flag, retreat_flag = battleAction(battle_flag, retreat_flag)
                        else:
                            None    
                    else: 
                        if(last_life_value - life_value > 20 or life_value <= 250):
                            battle_flag, retreat_flag = retreatAction(battle_flag, retreat_flag)
                        elif(life_value > 280):
                            battle_flag, retreat_flag = battleAction(battle_flag, retreat_flag)
                        else:
                            None
                    last_life_value = life_value
                    cur_time = time.time()
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  
    
