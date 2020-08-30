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
    logs_path 	= AI_BOTS_DIR+'/training_scripts/AI-models/dota2/lstm-logs/'
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
    model_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/dota2/frozen_inference_graph.pb'
    label_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/dota2/label_map.pbtxt'
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

def dotaBotActions():
    keyboard_action.mouse_click([1686, 1686, 1050, 1050])
    time.sleep(2)
    keyboard_action.mouse_click([1686, 1686, 1050, 1050])
    time.sleep(6)
    # random role
    # keyboard_action.mouse_click([1672, 1672, 816, 816])
    # choose hero luna
    keyboard_action.mouse_click([823, 823, 431, 431])
    time.sleep(2)
    keyboard_action.mouse_click([1481, 1481, 815, 815])
    time.sleep(6)
    # skip ahead
    keyboard_action.mouse_click([158, 158, 812, 812])
    time.sleep(4)
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

def gameBotParamInit():
    count 		= 0
    n_input 		= 1
    tensor_size 	= 2
    lstm_classes 	= 3
    n_hidden 		= 512
    lstm_based_Bots	= 1
    cnn_classes 	= 2
    return count, n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes

if __name__ == '__main__':
    player1Pos 		= [574, 574, 44, 44]
    player2Pos 		= [634, 634, 44, 44]
    player3Pos 		= [695, 695, 45, 45]
    player4Pos 		= [758, 758, 43, 43]
    player5Pos 		= [821, 821, 42, 42]
    commonPos 		= [678, 678, 1015, 1015]
    startPos 		= [38, 38, 1055, 1055]
    x_dim 		= range(786,1096,1)
    last_life_value 	= 0
    retreat_flag 	= 0
    battle_flag 	= 0
    training_data	= []
    collect_data	= 1

    count, n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = gameBotParamInit()
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode, terminalFocus = globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath, lstmInputVec 	= LSTMInit(AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= CNNInit(AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= logsInit(RESULT_DIR)
    commandInit("dota2", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode)
    dotaBotActions()

    file_name = '../training_data/dota2/raw-data/training_data' + str(int(time.time())) + '.npy'
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph=cnnDetection_graph) as cnnSession:
                start_time 	= time.time()
                cur_time 	= time.time()
                position_vec = [0, 1080]
                while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
                    gamescreen = cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    obj_set = set(obj_classes)
                    if 1 in obj_set:
                        hero_index = obj_classes.index(1)
                        pos_hero = obj_positions[hero_index]
                        position_vec = [int((pos_hero[0]+pos_hero[1])/2), int((pos_hero[0]+pos_hero[1])/2)]
                    life_value = 0
                    for x in x_dim:
                        if gamescreen[1045,x,1] > 100:
                            life_value+=1
                    if(last_life_value - life_value > 20 or life_value <= 250):
                        training_data.append([position_vec+[last_life_value, life_value],[0,0,1]])
                        battle_flag = 0
                        if(retreat_flag < 5):
                            keyboard_action.mouse_click([40,40,1044,1044],2,0)
                            retreat_flag += 1
                        elif(retreat_flag==5):
                            retreat_flag += 1
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
                            keyboard_action.mouse_click(player2Pos)
                            time.sleep(1)
                            keyboard_action.mouse_double_click(commonPos)
                            time.sleep(3)
                            keyboard_action.mouse_click(player3Pos)
                            time.sleep(1)
                            keyboard_action.mouse_double_click(commonPos)
                            time.sleep(3)
                            keyboard_action.mouse_click(player4Pos)
                            time.sleep(1)
                            keyboard_action.mouse_double_click(commonPos)
                            time.sleep(3)
                            keyboard_action.mouse_click(player5Pos)
                            time.sleep(1)
                            keyboard_action.mouse_double_click(commonPos)
                            time.sleep(3)
                            keyboard_action.mouse_click(player1Pos)
                            time.sleep(1)
                            keyboard_action.mouse_double_click(commonPos)
                    elif(life_value > 280):
                        training_data.append([position_vec+[last_life_value, life_value],[1,0,0]])
                        retreat_flag = 0
                        if(battle_flag < 5):
                            keyboard_action.mouse_click([115,115,990,990],2,0)
                            battle_flag += 1
                        else:
                            None
                    else:
                        training_data.append([position_vec+[last_life_value, life_value],[0,1,0]])
                        None

                    if collect_data ==1 and len(training_data) % 10 == 0:
                        print(len(training_data))
                        np.save(file_name, training_data)

                    last_life_value = life_value
                    time.sleep(3)
                    cur_time = time.time()
                output_file.close()  
    
