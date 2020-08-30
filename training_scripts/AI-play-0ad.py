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
from datetime import datetime
from TOD_Universal import TOD_Universal

def zeroADBotActions():
    keyboard_action.mouse_click(OK_position)
    time.sleep(1)
    keyboard_action.mouse_click(single_player_position)
    time.sleep(1)
    keyboard_action.mouse_click(load_game_position)
    time.sleep(1)
    keyboard_action.mouse_click(load_position)
    time.sleep(3)
    return

if __name__ == '__main__':
    stage_run = 0
    i_counter = 0
    station_index = 0
    collect_data  = 1

    OK_position = [827,827,734,734]
    single_player_position = [174,174,229,229]
    load_game_position = [409, 409, 291, 291]
    load_position=[996, 996, 844, 844]
    map_position = [236,236,859,859]
    enemy_center = [552,552,1054,1054]
    yes_position = [1048, 1048, 631, 631]
    continue_position = [1826, 1826, 1061, 1061]
    station_position = [[486,486,979,979],[550,550,1026,1026],[622,622,997,997],[567,567,941,941]]

    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = cgrAPI.gameBotParamInit(1, 2, 5, 512, 1, 19)
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode = cgrAPI.globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath 			= cgrAPI.LSTMInit("0ad", AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= cgrAPI.CNNInit("0ad", AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= cgrAPI.logsInit(RESULT_DIR)
    cgrAPI.commandInit("0ad", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode, 5)
    zeroADBotActions()

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph = cnnDetection_graph) as cnnSession:
                start_time = time.time()
                cur_time = time.time()
                last_cur_time   = cur_time
                while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
                    lstm_start	= 0
                    lstm_end	= 0
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    cv_start = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end   = time.time()
                    obj_set = set(obj_classes)
                    if 16 in obj_set or 17 in obj_set:
                        keyboard_action.mouse_click(yes_position)
                        time.sleep(1)
                        keyboard_action.mouse_click(continue_position)
                        time.sleep(1)
                        continue
                    obj_index = 0
                    lstm_input = [0, station_index]
                    if stage_run == 0:
                        if lstm_based_Bots == 1:
                            if 10 in obj_set:#deer
                                keyboard_action.tap_key('4')
                                deer_index = obj_classes.index(10)
                                lstm_input = [1, station_index]
                                obj_index  = deer_index
                            if 11 in obj_set:#tree
                                keyboard_action.tap_key('2')
                                tree_index = obj_classes.index(11)
                                lstm_input = [1, station_index]
                                obj_index  = tree_index
                            if 12 in obj_set:#bush
                                keyboard_action.tap_key('1')
                                bush_index = obj_classes.index(12)
                                lstm_input = [1, station_index]
                                obj_index  = bush_index
                            if 13 in obj_set:#stone
                                keyboard_action.tap_key('3')
                                stone_index = obj_classes.index(13)
                                lstm_input = [1, station_index]
                                obj_index  = stone_index
                            if i_counter%200 == 0:#knight
                                keyboard_action.tap_key('5', n=2)
                                lstmInputVec    = lstm_input
                                lstmRealInput   = np.reshape(lstmInputVec,[-1,n_input,tensor_size])
                                lstm_start 	= time.time()
                                onehot_pred     = lstmSession.run(lstmPred, feed_dict={lstmX: lstmRealInput})
                                lstm_end 	= time.time()
                                onehot_pred_index = np.argmax(onehot_pred)
                                if onehot_pred_index == 0:#retreat
                                    keyboard_action.mouse_click(obj_positions[obj_index],button=2)
                                elif onehot_pred_index == 1:#battle action
                                    keyboard_action.mouse_click(station_position[0], button=2, rand=0)
                                    station_index = 0
                                elif onehot_pred_index == 2:#retreat
                                    keyboard_action.mouse_click(station_position[1], button=2, rand=0)
                                    station_index = 1
                                elif onehot_pred_index == 3:#battle action
                                    keyboard_action.mouse_click(station_position[2], button=2, rand=0)
                                    station_index = 2
                                elif onehot_pred_index == 4:#battle action
                                    keyboard_action.mouse_click(station_position[3], button=2, rand=0)
                                    station_index = 3
                                else:
                                    None
                                if i_counter == 2000:
                                    stage_run = 1
                                    keyboard_action.tap_key('5', n=2)
                                    keyboard_action.mouse_click(enemy_center, button=2, rand=0)
                                    i_counter = 0
                            else:
                                keyboard_action.tap_key('5',n=2)
                            i_counter = i_counter+1
                            print(i_counter)
                        else:
                            if 10 in obj_set:#deer
                                keyboard_action.tap_key('4')
                                deer_index = obj_classes.index(10)
                                keyboard_action.mouse_click(obj_positions[deer_index],button=2)
                            if 11 in obj_set:#tree
                                keyboard_action.tap_key('2')
                                tree_index = obj_classes.index(11)
                                keyboard_action.mouse_click(obj_positions[tree_index],button=2)
                            if 12 in obj_set:#bush
                                keyboard_action.tap_key('1')
                                bush_index = obj_classes.index(12)
                                keyboard_action.mouse_click(obj_positions[bush_index],button=2)
                            if 13 in obj_set:#stone
                                keyboard_action.tap_key('3')
                                stone_index = obj_classes.index(13)
                                keyboard_action.mouse_click(obj_positions[stone_index],button=2)
                            if i_counter%200 == 0:#knight
                                keyboard_action.tap_key('5', n=2)
                                station_index =(station_index + 1)%4
                                keyboard_action.mouse_click(station_position[0], button=2, rand=0)
                                if i_counter == 2000:
                                    stage_run = 1
                                    keyboard_action.tap_key('5', n=2)
                                    keyboard_action.mouse_click(enemy_center, button=2, rand=0)
                                    i_counter = 0
                            else:
                                keyboard_action.tap_key('5',n=2)
                            i_counter = i_counter+1
                            print(i_counter)
                    elif stage_run == 1:
                        i_counter = i_counter+1
                        print(i_counter)
                        if i_counter == 400:
                            stage_run = 2
                        if i_counter%250 == 0:
                            keyboard_action.tap_key('5', n=2)
                            continue
                        if i_counter%200 == 0:
                            keyboard_action.tap_key('1', n=2)
                            continue
                        if i_counter%150 == 0:
                            keyboard_action.tap_key('2', n=2)
                            continue
                        if i_counter%100 == 0:
                            keyboard_action.tap_key('3', n=2)
                            continue
                        if i_counter%50 == 0:
                            keyboard_action.tap_key('4', n=2)
                            continue
                    else:
                        keyboard_action.mouse_click(yes_position)
                        time.sleep(1)
                        keyboard_action.mouse_click(continue_position)
                        time.sleep(1)
                        zeroADBotActions()
                        station_index = 0
                        i_counter = 0
                        stage_run = 0
                        continue
                    cur_time = time.time()   
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+" "+str((cur_time - last_cur_time)*1000)+"\n"
                    output_file.write(row)
                    last_cur_time = cur_time
                output_file.close()  
         

