import numpy as np
import tensorflow as tf
import os
import cv2
import mss
import keyboard_action
import random
import time
import tools
from tensorflow.contrib import rnn
from TOD import TOD


#frequently modified parameters
logs_path = './AI-models/supertuxkart/lstm-logs/'
#logs_path = '../training_data/logs/'
n_input = 1 		# sequential input vector numbers for lstm
tensor_size = 6         # length of each input vector
n_classes = 3           # number of classes for the output
#feature_mode = 0
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

# Launch the graph
all_stops = [[120,802],[128,723],[72,688],[76,560],[128,540],[168,595],[166,667],[206,732]]
#all_stops = [[100,670],[112,580-20],[70,540],[70,420],[115,416],[165-20,461],[182-35,505],[180-30,585]]
#all_shapes = [100,670,112,580-20,70,540,70,420,115,416,165,461,182,505,180-30,585]
all_shape  = [120,802,128,723,72,688,76,560,128,540,168,595,166,667,206,732]
last_position = [112,900]
#last_position = [100,800]
profile_position_before = [112,900]
#profile_position_before = [100,800]
last_time = time.time()
with mss.mss(display=':0.0') as sct:
    detector = TOD(sct)
    detection_graph = detector._load_model()
    start = 0
    iteration_index = 0
    with tf.Session() as session:
        session.run(init)
        if os.path.isfile(logs_path+"checkpoint"):
            saver.restore(session,logs_path+"lstm-model")
        with tf.Session(graph=detection_graph) as sess:
        #figure_vector:[[3 points][ angle ][10 points]]
            while True:
                if feature_mode == 2: # 10 points
                    screen = np.array(sct.grab(detector.region))
                    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                    figure,image_show = detector.detect(screen, detection_graph, sess)
                    figure_vector = np.reshape((all_shapes + last_position + figure[-2:]),(-1,n_input,tensor_size))
                    last_position = figure[-2:]
                elif feature_mode == 0: # 3 points
                    print('status:',start)
                    start, figure_vector, cur,image_show = detector.generate_feature(start, detection_graph, sess)
                    figure_vector = np.reshape(figure_vector[feature_mode],[-1,n_input,tensor_size])
                    print(figure_vector)
                elif feature_mode == 1: # angles
                    print('status:',start)
                    start, figure_vector,cur,image_show = detector.generate_feature(start, detection_graph, sess)
                    figure_vector = np.reshape(figure_vector[feature_mode],[-1,n_input,tensor_size])
                elif feature_mode == 3: # 3 points + objectDetection
                    print('status:',start)
                    start, figure_vector, cur, obj_positions, image_show = detector.generate_feature_objects(start, detection_graph, sess)
                    figure_vector = np.reshape(figure_vector[0],[-1,n_input,tensor_size])
                    print(figure_vector)
                    #print(obj_positions)
                else:
                    print('other mode')
                    break
                iteration_index +=1
                if iteration_index % 6 == 0:
                    #if detector.samePosition(cur_position=cur):
                    if detector.samePositionAsBefore(cur_position=cur):
                        keyboard_action.move_back()
                        continue
                    cur_time = time.time()
                    print('time: ',cur_time-last_time)
                    last_time = cur_time
                    figure_vector[0][0][2:4] = profile_position_before
                    iteration_index = 0
                    onehot_pred = session.run(pred, feed_dict={x: figure_vector})
                    onehot_pred_index = np.argmax(onehot_pred)
                    profile_position_before = figure_vector[0][0][4:]
                    print("profile_before:", profile_position_before)
                    if onehot_pred_index == 0:
                        keyboard_action.turn_left()
                        print('nornal left')
                    elif onehot_pred_index == 2:
                        keyboard_action.turn_right()
                        print('nornal right')
                    else: 
                        keyboard_action.move_up()
                        print('nornal up')
                elif feature_mode == 3:
                    if 1==tools.boudOverlap(obj_positions):
                        keyboard_action.quick_left()
                        print('re-ajust left')
                    elif 2==tools.boudOverlap(obj_positions):
                        keyboard_action.quick_right()
                        print('re-ajust right')
                    else:
                        keyboard_action.move_up()
                        print('re-ajust up')
                else:
                    None
                for i in range(8):
                    cv2.circle(image_show,tuple(all_stops[i]),25,(255,0,0),1) 
                cv2.imshow('detection', image_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()



   
