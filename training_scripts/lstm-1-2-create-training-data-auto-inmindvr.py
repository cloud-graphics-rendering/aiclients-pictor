import sys
import os
import time
import tools
import cv2
import mss
import numpy as np
import keyboard_action
import keyboard
import pyautogui
import tensorflow as tf
from tensorflow.contrib import rnn
from pykeyboard import PyKeyboard
from pymouse import PyMouse
from datetime import datetime
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
    #pyautogui.moveTo(inmind_center[0], inmind_center[1])
    time.sleep(5)
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
    logs_path 	= AI_BOTS_DIR+'/training_scripts/AI-models/inmindvr/lstm-logs/'
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
    model_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/inmindvr/frozen_inference_graph.pb'
    label_dir 	= AI_BOTS_DIR+'/training_scripts/AI-models/inmindvr/label_map.pbtxt'
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
    tensor_size 	= 20
    lstm_classes 	= 5
    n_hidden 		= 512
    lstm_based_Bots	= 1
    cnn_classes 	= 1
    return n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes

def centerOutofBox(region, point):
    if point[0] > region[1] or point[0] < region[0] or point[1] < region[2] or point[1] > region[3]:
        return True
    else:
        return False

if __name__ == '__main__':
    inmind_center = [940, 566]#960,564
    i_counter = 0
    last_drag = 0
    output_keysList = []
    training_data = []
    file_name = '../training_data/inmindvr/training_data' + str(int(time.time())) + '.npy'
    n_input, tensor_size, lstm_classes, n_hidden, lstm_based_Bots, cnn_classes = gameBotParamInit()
    associate_flag, RUNNING_TIME, AI_BOTS_DIR, RESULT_DIR, BIND_CPU, HUMAN_RUN, Reso_Width, Reso_Hight, MultipleMode, terminalFocus = globalParamInit()
    lstmX, lstmPred, lstmInit, lstmSaver, lstmLogPath, lstmInputVec 	= LSTMInit(AI_BOTS_DIR, n_input, tensor_size, lstm_classes, n_hidden)
    pic_region, cnnDetection_graph, cnnDetector 			= CNNInit(AI_BOTS_DIR, Reso_Width, Reso_Hight, cnn_classes)
    output_file	= logsInit(RESULT_DIR)
    commandInit("inmindvr", associate_flag, RUNNING_TIME, BIND_CPU, HUMAN_RUN, MultipleMode)

    with mss.mss(display=':0.0') as sct:
        with tf.Session() as lstmSession:
            lstmSession.run(lstmInit)
            if os.path.isfile(lstmLogPath+"checkpoint"):
                lstmSaver.restore(lstmSession,lstmLogPath+"lstm-model")
            with tf.Session(graph = cnnDetection_graph) as cnnSession:
                start_time = time.time()
                cur_time = time.time()
                while (cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0):
                    pyautogui.moveTo(inmind_center[0], inmind_center[1])
                    i_counter = i_counter+1
                    #print("i_counter:",i_counter)
                    gamescreen 	= cv2.cvtColor(np.array(sct.grab(pic_region)), cv2.COLOR_BGR2RGB)
                    lstm_start	= 0
                    lstm_end	= 0
                    cv_start = time.time()
                    obj_classes, obj_positions, image_show = cnnDetector.detect_objects(gamescreen, cnnDetection_graph, cnnSession)
                    cv_end   = time.time()
                    obj_set = set(obj_classes)
                    if 1 in obj_set: #start
                        map_index = obj_classes.index(1)
                        map_position = obj_positions[map_index]
                        if centerOutofBox(map_position, inmind_center):
                            # drag 1 to center
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                    elif 3 in obj_set: #again
                        map_index = obj_classes.index(3)
                        map_position = obj_positions[map_index]
                        # drag 3 to center
                        if centerOutofBox(map_position, inmind_center):
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                            if i_counter > 1000:
                                pyautogui.moveTo(inmind_center[0], inmind_center[1])
                                i_counter = 0
                    elif 2 in obj_set: #web.com
                        map_index = obj_classes.index(2)
                        map_position = obj_positions[map_index]
                        # drag 3 to center
                        if centerOutofBox(map_position, inmind_center):
                            newPos = [(map_position[0]+map_position[1])/2, (map_position[2]+map_position[3])/2]
                            diffx = newPos[0]-inmind_center[0]
                            diffy = newPos[1]-inmind_center[1]+80
                            pyautogui.dragRel(diffx, diffy,duration = 0.2)
                            time.sleep(3)
                            pyautogui.dragRel(-diffx, -diffy,duration = 0.2)
                            if i_counter > 1000:
                                pyautogui.moveTo(inmind_center[0], inmind_center[1])
                                i_counter = 0
                    else:
                        lower = np.array([171,0,21], dtype = "uint8")
                        upper = np.array([255,44,100], dtype = "uint8")
                        mask = cv2.inRange(gamescreen, lower, upper)
                        output = cv2.bitwise_and(gamescreen, gamescreen, mask = mask)
                        img1gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
                        retval1, img1bin = cv2.threshold(img1gray,50,255,0)
                        el     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        e_img  = cv2.erode(img1bin, el, iterations=1)
                        d_img  = cv2.dilate(e_img, el, iterations=20)
                        d_img,contours,hierarchy = cv2.findContours(d_img, 1, 2)
                        centerList = []
                        distList   = []
                        for contour in contours:
                            m_img  = cv2.moments(contour)
                            center = (int(m_img['m10']/m_img['m00']),int(m_img['m01']/m_img['m00']))
                            centerList.append(center)
                            distList.append(tools.pointsdist(center, inmind_center))
                        if len(centerList) != 0:
                            center_closest = centerList[np.argmax(distList)]
                            pyautogui.dragRel(int(center_closest[0]-inmind_center[0]), int(center_closest[1]-inmind_center[1]),duration = 0.5)
                            time.sleep(1.5)
                            training_data.append([[1,last_drag],[1,0,0]])#drag to cell
                        elif i_counter%2==0:
                            pyautogui.dragRel(0, -150,duration = 0.2)
                            pyautogui.dragRel(0, 150,duration = 0.2)
                            training_data.append([[0,last_drag],[0,1,0]])#drag up
                            last_drag = 0
                        else:
                            pyautogui.dragRel(0, 150,duration = 0.2)
                            pyautogui.dragRel(0, -150,duration = 0.2)
                            training_data.append([[0,last_drag],[0,0,1]])#drag down
                            last_drag = 1
                        if len(training_data) % 100 == 0:
                            print(len(training_data))
                            np.save(file_name, training_data)
                    cur_time = time.time()   
                    row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((lstm_end-lstm_start)*1000)+"\n"
                    output_file.write(row)
                output_file.close()  
