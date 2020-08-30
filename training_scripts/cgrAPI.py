import math
import sys
import os
import time
import cv2
import mss
import numpy as np
import keyboard_action
import keyboard
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from datetime import datetime
from TOD_Universal import TOD_Universal

def boudOverlap(obj_positions):
   if len(obj_positions)<3 or obj_positions[1]==[560,660,360,580]:
       return 0
   rider = obj_positions[1]
   positions = obj_positions[2:]
   turn_left = 0
   turn_right = 0
   for x in positions:
       if rider[2] < x[3] and rider[2] > x[2]:
           diff = (x[0]+x[1])/2.0 - (rider[0]+rider[1])/2.0
           if diff>0 and diff <150:
              turn_left += 1;
           elif diff < 0 and diff >-150:
               turn_right += 1;
           else:
               None
       else:
           None
   if turn_left == 0 and turn_right !=0:
       return 2
   elif turn_right == 0 and turn_left != 0:
       return 1
   else:
       return 0 

def pointsdist(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def gameBotParamInit(n_input,tensor_size,lstm_classes,n_hidden,lstm_based_Bots,cnn_classes):
    return n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes

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
    #terminalFocus	= [200,200,200,200]
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
    return associate_flag,RUNNING_TIME,AI_BOTS_DIR,RESULT_DIR,BIND_CPU,HUMAN_RUN,Reso_Width,Reso_Hight,MultipleMode

def commandInit(appName, associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode, delay):
    keyboard_action.mouse_click([960,960,540,540])
    time.sleep(1)
    key_board = PyKeyboard()
    key_board.type_string('cd $CGR_BENCHMARK_PATH/')
    key_board.tap_key(key_board.enter_key)
    key_board.type_string('./collectData.sh'+' '+str(appName)+' '+str(associate_flag)+' '+str(RUNNING_TIME)+' '+str(BIND_CPU)+' '+str(HUMAN_RUN)+' '+str(MultipleMode)+' &')
    key_board.tap_key(key_board.enter_key)
    RUNNING_TIME -= 30
    time.sleep(delay)
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

def LSTMInit(appName, AI_BOTS_DIR,n_input,tensor_size,n_classes,n_hidden):
    '''
    n_input 	= 1 		# sequential input vector numbers for lstm
    tensor_size = 6         	# length of each input vector
    n_classes 	= 3           	# number of classes for the output
    n_hidden 	= 512    	# number of units in RNN cell
    '''
    logs_path 	= AI_BOTS_DIR+'/training_scripts/AI-models/'+appName+'/lstm-logs/'
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
    
    return x, pred, init, saver, logs_path

def CNNInit(appName, AI_BOTS_DIR, Reso_Width, Reso_Hight, num_class):
    model_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/'+appName+'/frozen_inference_graph.pb'
    label_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/'+appName+'/label_map.pbtxt'
    region	= {'top': 0, 'left': 0, 'width': Reso_Width, 'height': Reso_Hight}
    detector 	= TOD_Universal(model_dir, label_dir, num_class, region)
    graph 	= detector._load_model()
    return region, graph, detector

def logsInit(resultPath):
    FILE_NAME = resultPath+'/cv_action_time.csv'
    output_file = open(FILE_NAME,"w")
    columnTitleRow = "DATE TIME CV_TIME LSTM_TIME CYCLE_TIME\n"
    output_file.write(columnTitleRow)
    return output_file


