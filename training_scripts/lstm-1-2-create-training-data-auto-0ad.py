import sys
import os
import time
import tools
import cv2
import mss
import numpy as np
import keyboard_action
import keyboard
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from datetime import datetime
from PIL import Image
from TOD_Universal import TOD_Universal

def globalParamInit():
    associate_flag 	= 0
    RUNNING_TIME	= 300
    AI_BOTS_DIR	        = ""
    RESULT_DIR  	= ""
    BIND_CPU  	        = 0
    HUMAN_RUN  	        = 0
    Reso_Width          = 1920
    Reso_Hight          = 1080
    MultipleMode        = 1
    terminalFocus	= [200,200,200,200]
    if(len(sys.argv) > 1):
        associate_flag 	= int(sys.argv[1])
        RUNNING_TIME	= int(sys.argv[2])
        AI_BOTS_DIR	= sys.argv[3]
        RESULT_DIR  	= sys.argv[4]
        BIND_CPU  	= int(sys.argv[5])
        HUMAN_RUN  	= int(sys.argv[6])
        Reso_Width      = int(sys.argv[7])
        Reso_Hight      = int(sys.argv[8])
        MultipleMode    = int(sys.argv[9])
    return associate_flag,RUNNING_TIME,AI_BOTS_DIR,RESULT_DIR,BIND_CPU,HUMAN_RUN,Reso_Width,Reso_Hight,MultipleMode,terminalFocus

def commandInit(appName, associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode):
    keyboard_action.mouse_click(terminalFocus)
    time.sleep(1)
    key_board = PyKeyboard()
    key_board.type_string('cd $CGR_BENCHMARK_PATH/')
    key_board.tap_key(key_board.enter_key)
    key_board.type_string('./collectData.sh'+' '+str(appName)+' '+str(associate_flag)+' '+str(RUNNING_TIME)+' '+str(BIND_CPU)+' '+str(HUMAN_RUN)+' '+str(MultipleMode)+' &')
    key_board.tap_key(key_board.enter_key)
    RUNNING_TIME -= 30
    time.sleep(10)
    return

def RNN(x, weights, biases, n_input, n_hidden):
    x = tf.unstack(x,n_input,1)
    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden,forget_bias=1.0),rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)])
    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def LSTMInit(AI_BOTS_DIR,n_input,tensor_size,n_classes,n_hidden):
    '''
    n_input 	= 1 		# sequential input vector numbers for lstm
    tensor_size = 6         	# length of each input vector
    n_classes 	= 3           	# number of classes for the output
    n_hidden 	= 512    	# number of units in RNN cell
    '''
    lstmInputVec = [0, 0]
    logs_path 	= AI_BOTS_DIR+'/training_scripts/AI-models/0ad/lstm-logs/'
    x = tf.placeholder("float", [None, n_input, tensor_size])	# tf Graph input
    weights = { 		# RNN output node weights and biases
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    pred = RNN(x, weights, biases, n_input, n_hidden)
    init = tf.global_variables_initializer() 	# Initializing the variables
    saver = tf.train.Saver() 			#generate saver
    
    return x, pred, init, saver, logs_path, lstmInputVec

def CNNInit(AI_BOTS_DIR, Reso_Width, Reso_Hight, num_class):
    model_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/0ad/frozen_inference_graph.pb'
    label_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/0ad/label_map.pbtxt'
    region	= {'top': 0, 'left': 0, 'width': Reso_Width, 'height': Reso_Hight}
    detector 	= TOD_Universal(model_dir, label_dir, num_class, region)
    graph 	= detector._load_model()
    return region, graph, detector

def logsInit(resultPath):
    FILE_NAME = resultPath+'/cv_action_time.csv'
    output_file = open(FILE_NAME,"w")
    columnTitleRow = "DATE TIME CV_TIME ACTION_TIME\n"
    output_file.write(columnTitleRow)
    return output_file

def gameBotParamInit():
    n_input 		= 1
    tensor_size 	= 3
    lstm_classes 	= 5
    n_hidden 		= 512
    lstm_based_Bots	= 1
    cnn_classes 	= 19
    return n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes

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

    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = gameBotParamInit()
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode, terminalFocus = globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath, lstmInputVec 	= LSTMInit(AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= CNNInit(AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= logsInit(RESULT_DIR)
    commandInit("0ad", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode)
    zeroADBotActions()

    file_name = '../training_data/0ad/raw-data/training_data' + str(int(time.time())) + '.npy'
    training_data = []

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph = cnnDetection_graph) as cnnSession:
                start_time = time.time()
                cur_time = time.time()
                while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    lstm_start	= 0
                    lstm_end	= 0
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
                        
                    if stage_run == 0:
                        if i_counter%200 == 0:#knight
                            keyboard_action.tap_key('5', n=2)
                            last_station_index = station_index
                            station_index =(station_index + 1)%4
                            keyboard_action.mouse_click(station_position[station_index], button=2, rand=0)
                            if i_counter == 2000:
                                stage_run = 1
                                keyboard_action.tap_key('5', n=2)
                                keyboard_action.mouse_click(enemy_center, button=2, rand=0)
                                i_counter = 0
                        else:
                            keyboard_action.tap_key('5',n=2)
                        result_list = [0,0,0,0,0]
                        result_list[station_index+1] = 1
                        data_record = [[0, last_station_index],result_list]
                        if 10 in obj_set:#deer
                            print('hunter ready')
                            keyboard_action.tap_key('4')
                            deer_index = obj_classes.index(10)
                            keyboard_action.mouse_click(obj_positions[deer_index],button=2)
                            data_record = [[1, last_station_index],[1,0,0,0,0]]
                        if 11 in obj_set:#tree
                            print('hoplite ready')
                            keyboard_action.tap_key('2')
                            tree_index = obj_classes.index(11)
                            keyboard_action.mouse_click(obj_positions[tree_index],button=2)
                            data_record = [[1, last_station_index],[1,0,0,0,0]]
                        if 12 in obj_set:#bush
                            print('famer ready')
                            keyboard_action.tap_key('1')
                            bush_index = obj_classes.index(12)
                            keyboard_action.mouse_click(obj_positions[bush_index],button=2)
                            data_record = [[1, last_station_index],[1,0,0,0,0]]
                        if 13 in obj_set:#stone
                            print('militia ready')
                            keyboard_action.tap_key('3')
                            stone_index = obj_classes.index(13)
                            keyboard_action.mouse_click(obj_positions[stone_index],button=2)
                            data_record = [[1, last_station_index],[1,0,0,0,0]]
                        i_counter = i_counter+1
                        print(i_counter)
                        training_data.append(data_record)
                        print(data_record)
                        if collect_data ==1 and len(training_data) % 10 == 0:
                            np.save(file_name, training_data)
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
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((cur_time-cv_end)*1000)+"\n"
                    output_file.write(row)
                output_file.close()  
         

