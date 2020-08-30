import numpy as np
import tensorflow as tf
import os
import cv2
import mss
import keyboard_action
import random
import time
import cgrAPI
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from TOD import TOD
from datetime import datetime
import sys

associate_flag 	= 0
RUNNING_TIME	= 300
AI_BOTS_DIR	= ""
RESULT_DIR  	= ""
BIND_CPU  	= 0
HUMAN_RUN  	= 0
Reso_Width      = 1960
Reso_Hight      = 1080
if(len(sys.argv) > 1):
    associate_flag 	= int(sys.argv[1])
    RUNNING_TIME	= int(sys.argv[2])
    AI_BOTS_DIR		= sys.argv[3]
    RESULT_DIR  	= sys.argv[4]
    BIND_CPU  		= int(sys.argv[5])
    HUMAN_RUN  		= int(sys.argv[6])
    Reso_Width          = int(sys.argv[7])
    Reso_Hight          = int(sys.argv[8])
    MultipleMode	= int(sys.argv[9])
#terminalFocus=[200,200,200,200]
terminalFocus=[980,980,540,540]
keyboard_action.mouse_click(terminalFocus)
time.sleep(1)
key_board = PyKeyboard()
key_board.type_string('cd $CGR_BENCHMARK_PATH/')
key_board.tap_key(key_board.enter_key)
key_board.type_string('./collectData.sh supertuxkart-1 '+str(associate_flag)+' '+str(RUNNING_TIME)+' '+str(BIND_CPU)+' '+str(HUMAN_RUN)+' '+str(MultipleMode)+' &')
key_board.tap_key(key_board.enter_key)
time.sleep(5)
key_board.tap_key(key_board.enter_key)
time.sleep(3)
RUNNING_TIME -= 30

#frequently modified parameters
logs_path = AI_BOTS_DIR+'/training_scripts/AI-models/supertuxkart-1/lstm-logs/'
n_input = 1 		# sequential input vector numbers for lstm
tensor_size = 6         # length of each input vector
n_classes = 3           # number of classes for the output
feature_mode = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, tensor_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden,forget_bias=1.0),rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)])
    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    #rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Initializing the variables
init = tf.global_variables_initializer()

#generate saver
saver = tf.train.Saver() #generate saver

### For a specific game ###
model_dir = AI_BOTS_DIR+'/training_scripts/AI-models/supertuxkart-1/frozen_inference_graph.pb'
label_dir = AI_BOTS_DIR+'/training_scripts/AI-models/supertuxkart-1/label_map.pbtxt'
num_class = 13
#pic_region = {'top': 0, 'left': 0, 'width': 1280, 'height': 960}
pic_region = {'top': 0, 'left': 0, 'width': 1960, 'height': 1080}

FILE_NAME = RESULT_DIR+'/supertuxkart_cv_action_time.csv'
output_file = open(FILE_NAME,"w")
columnTitleRow = "TIME, CV_TIME, ACTION_TIME\n"
output_file.write(columnTitleRow)

# Launch the graph
#all_stops = [[118,948],[134,872],[76,786],[70,680],[132,645],[169,702],[159,764],[208,824]]
all_stops = [[140,1000],[145,845],[98,814],[90,680],[130,645],[180,670],[184,750],[200,850]]
#all_stops = []

#for point in all_stop:
#    point[0]=point[0]*Reso_Width/1280
#    point[1]=point[1]*Reso_Hight/960
#    all_stops.append([point[0],point[1]])

all_shape  = np.reshape(all_stops, 16)
last_position = all_stops[0]
profile_position_before = [127,1045]
with mss.mss(display=':0.0') as sct:
    detector = TOD(sct, model_dir, label_dir, num_class, pic_region, all_stops, all_shape, last_position)
    detection_graph = detector._load_model()
    start = 0
    iteration_index = 0
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    with tf.Session(config=config) as session:
        session.run(init)
        if os.path.isfile(logs_path+"checkpoint"):
            saver.restore(session,logs_path+"lstm-model")
        with tf.Session(graph=detection_graph, config=config) as sess:
        #figure_vector:[[3 points][ angle ][10 points]]
            start_time = time.time()
            cur_time = time.time()
            cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("detection", 1960, 1080)
            cv2.moveWindow("detection", 1960,0)
            missing_count = 0
            keyboard_action.mouse_click([1666,1666,995,995])
            while ((iteration_index <= 12) and (HUMAN_RUN==0)):
                keyboard_action.go_up()
                iteration_index +=1
                
            while ((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
                #keyboard_action.mouse_click([1666,1666,995,995])
                cv_start = time.time()
                print('status:',start)
                start, figure_vector, cur, obj_positions, image_show = detector.generate_feature_objects(start, detection_graph, sess)
                figure_vector = np.reshape(figure_vector[0],[-1,n_input,tensor_size])

                iteration_index +=1
                cv_end   = time.time()
                if iteration_index % 2 == 0:
                    if detector.samePositionAsBefore(cur_position=cur):
                        missing_count +=1
                        if missing_count % 8 == 0:
                            keyboard_action.rescue()
                            missing_count = 0
                        else:
                            keyboard_action.move_up()
                        #keyboard_action.move_back()
                        #continue
                    else:
                        missing_count = 0
                    cur_time = time.time()
                    figure_vector[0][0][2:4] = profile_position_before
                    iteration_index = 0
                    onehot_pred = session.run(pred, feed_dict={x: figure_vector})
                    
                    onehot_pred_index = np.argmax(onehot_pred)
                    profile_position_before = figure_vector[0][0][4:]
                    if onehot_pred_index == 0:
                        keyboard_action.turn_left()
                    elif onehot_pred_index == 2:
                        keyboard_action.turn_right()
                    else: 
                        keyboard_action.move_up()
                elif feature_mode == 3:
                    if 1==cgrAPI.boudOverlap(obj_positions):
                        keyboard_action.quick_left()
                    elif 2==cgrAPI.boudOverlap(obj_positions):
                        keyboard_action.quick_right()
                    else:
                        keyboard_action.move_up()
                if(associate_flag == 0):
                    start_time = time.time()
                cur_time = time.time()
                row = str(datetime.now())+","+str(cv_end-cv_start)+","+str(cur_time-cv_end)+"\n"
                output_file.write(row)
                for point in all_stops:
                    point = tuple(point)
                    cv2.circle(image_show, point, 10, (255,0,0), 2)
                cv2.imshow('detection', image_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            output_file.close()  
