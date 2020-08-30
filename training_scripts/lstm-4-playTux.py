# -*- coding: utf-8 -*-
# lstm_playTux.py

import numpy as np
import cv2
import time
import os
import mss
import keyboard_action
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

all_stops = [[100,670],[112,580],[70,540],[70,420],[115,416],[165,461],[182,505],[180,585]]
pixel_error = 15
method = 1#analyse
stuck_count = 0
stuck_threshold = 4

play = 1
collect_data = 1
good_angle = 40
figure_vector = []
training_data=[]
last_position=[100,680]
start = 0
dst = all_stops[1]
angle=0
feature = []

#def get_model_updown():
    # Network building
#    net = tflearn.input_data(shape=[None, 4, 2], name='net1_layer1')
#    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net1_layer2')
#    net = tflearn.dropout(net, 0.6, name='net1_layer3')
#    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net1_layer4')
#    net = tflearn.dropout(net, 0.6, name='net1_layer5')
#    net = tflearn.fully_connected(net, 3, activation='softmax', name='net1_layer6')
#    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
                             #name='net1_layer7')
#    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)


#def get_model_leftright():
    # Network building
#    net = tflearn.input_data(shape=[None, 4, 2], name='net2_layer1')
#    net = tflearn.lstm(net, n_units=256, return_seq=True, name='net2_layer2')
#    net = tflearn.dropout(net, 0.6, name='net2_layer3')
#    net = tflearn.lstm(net, n_units=256, return_seq=False, name='net2_layer4')
#    net = tflearn.dropout(net, 0.6, name='net2_layer5')
#    net = tflearn.fully_connected(net, 3, activation='softmax', name='net2_layer6')
#    net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.001,
#                             name='net2_layer7')
#    return tflearn.DNN(net, clip_gradients=5.0, tensorboard_verbose=0)

def check_stops(position):
    index=-1
    for stop in all_stops:
        index = index+1
        if position[0] > stop[0]-pixel_error and position[0] < stop[0] + pixel_error and position[1] > stop[1] - pixel_error and position[1] < stop[1] + pixel_error:
            return index
    return 10        

def samePosition(last_position=[0,0], cur_position=[1,1]):
    global stuck_count
    global stuck_threshold
    print("stuck_count:",stuck_count)
    if abs(last_position[0]-cur_position[0])<1 and abs(last_position[1]-cur_position[1])<1:
        stuck_count +=1
    else:
        stuck_count = 0
    if(stuck_count > stuck_threshold):
        stuck_count = 0
        return True
    else:
        return False

def calculate_angle(last=[0,0], cur=[0,1], dst=[1,0]):
    v0=np.array(cur)-np.array(last)
    v1=np.array(dst)-np.array(last)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def gotoDestination(start):
   global dst
   global last_position
   global good_angle
   global angle
   global feature
 
   screen = np.array(sct.grab(region))
   screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
   figure = detector.detect(screen)
   angle = calculate_angle(last_position,figure[-2:],dst)
   if start + 1 == 8:
       target_num = 1
   else:
       target_num = start+1

   if(check_stops(figure[-2:])!=target_num):
       while(abs(angle) > good_angle):
           if angle > good_angle:
               if samePosition(last_position, figure[-2:]):
                   if collect_data == 1:
                       print([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
                       training_data.append([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
                   keyboard_action.move_back()
                   return start
               if collect_data == 1:
                   print([[dst + last_position + figure[-2:]],[angle],[0,0,0,1]])
                   training_data.append([[dst + last_position + figure[-2:]],[angle],[0,0,0,1]])
               keyboard_action.turn_right()
           elif angle < -good_angle:
               if samePosition(last_position, figure[-2:]):
                   if collect_data == 1:
                       print([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
                       training_data.append([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
                       keyboard_action.move_back()
                   return start
               if collect_data == 1:
                   print([[dst + last_position + figure[-2:]],[angle],[1,0,0,0]])
                   training_data.append([[dst + last_position + figure[-2:]],[angle],[1,0,0,0]])
               keyboard_action.turn_left()
           else:
               return start
           last_position = figure[-2:]
           screen = np.array(sct.grab(region))
           screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
           figure = detector.detect(screen)
           angle = calculate_angle(last_position,figure[-2:],dst)

       if samePosition(last_position, figure[-2:]):
           if collect_data == 1:
               print([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
               training_data.append([[dst + last_position + figure[-2:]],[angle],[0,0,1,0]])
           keyboard_action.move_back()
           return start
       if collect_data == 1:
           print([[dst + last_position + figure[-2:]],[angle],[0,1,0,0]])
           training_data.append([[dst + last_position + figure[-2:]],[angle],[0,1,0,0]])
       keyboard_action.move_up()
   else:
       start = start+1
       if start == 7:
           dst = all_stops[1]
       elif start ==8:
           start = 1
           dst = all_stops[start+1]
       else:
           dst = all_stops[start+1]
           
       if angle > good_angle:
           if collect_data == 1:
               print([[dst + last_position + figure[-2:]],[angle],[0,0,0,1]])
               training_data.append([[dst + last_position + figure[-2:]],[angle],[0,0,0,1]])
           keyboard_action.turn_right()
       elif angle < -good_angle:
           if collect_data == 1:
               print([[dst + last_position + figure[-2:]],[angle],[1,0,0,0]])
               training_data.append([[dst + last_position + figure[-2:]],[angle],[1,0,0,0]])
           keyboard_action.turn_left()
       else:
           if collect_data == 1:
               print([[dst + last_position + figure[-2:]],[angle],[0,1,0,0]])
               training_data.append([[dst + last_position + figure[-2:]],[angle],[0,1,0,0]])
           keyboard_action.move_up()
   last_position = figure[-2:]

   return start

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/workspace/bench-train-games/superTuxCNN-models/frozen_inference_graph.pb'
        self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/label_map.pbtxt'
        self.NUM_CLASSES 	= 1
        self.detection_graph 	= self._load_model()
        self.category_index 	= self._load_label_map()
        self.shape   = []

    def _load_model(self):
        detection_graph 	= tf.Graph()
        with detection_graph.as_default():
            od_graph_def	= tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map	= label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories	= label_map_util.convert_label_map_to_categories(label_map,max_num_classes=self.NUM_CLASSES,use_display_name=True)
        category_index	= label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor	  = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes		  = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores		  = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes		  = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections	  = self.detection_graph.get_tensor_by_name('num_detections:0')
                

                (boxes, scores, classes, num_detections) = sess.run(
                       [boxes, scores, classes, num_detections],
                       feed_dict={image_tensor: image_np_expanded})
                box_centers=vis_util.visualize_realboxes_and_labels_on_image_array(
                       image,
                       np.squeeze(boxes),
                       np.squeeze(classes).astype(np.int32),\
                       np.squeeze(scores),
                       self.category_index,
                       use_normalized_coordinates=True,
                       line_thickness = 8)
                if len(box_centers) > 0:
                    feature_vector = (self.shape + box_centers[0])
                else:
                    feature_vector = (self.shape + [350,500])
                print(feature_vector)
                if method == 1:
                    cv2.rectangle(image,(all_stops[0][0]-pixel_error,all_stops[0][1]-pixel_error),(all_stops[0][0]+pixel_error,all_stops[0][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[1][0]-pixel_error,all_stops[1][1]-pixel_error),(all_stops[1][0]+pixel_error,all_stops[1][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[2][0]-pixel_error,all_stops[2][1]-pixel_error),(all_stops[2][0]+pixel_error,all_stops[2][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[3][0]-pixel_error,all_stops[3][1]-pixel_error),(all_stops[3][0]+pixel_error,all_stops[3][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[4][0]-pixel_error,all_stops[4][1]-pixel_error),(all_stops[4][0]+pixel_error,all_stops[4][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[5][0]-pixel_error,all_stops[5][1]-pixel_error),(all_stops[5][0]+pixel_error,all_stops[5][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[6][0]-pixel_error,all_stops[6][1]-pixel_error),(all_stops[6][0]+pixel_error,all_stops[6][1]+pixel_error),(0,0,0))
                    cv2.rectangle(image,(all_stops[7][0]-pixel_error,all_stops[7][1]-pixel_error),(all_stops[7][0]+pixel_error,all_stops[7][1]+pixel_error),(0,0,0))
                cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                cv2.imshow("detection", image)
                return feature_vector

if __name__ == '__main__':
    
    file_name = 'rnn6-16/training_data' + str(int(time.time())) + '.npy'
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    with mss.mss(display=':0.0') as sct:
        region={'top': 60, 'left': 64, 'width': 1020, 'height': 730}
        detector = TOD()
        while(True):
            #last_time=time.time()
            if method == 1:
                #analysis
                print('start:',start)
                start = gotoDestination(start)

                if collect_data ==1 and len(training_data) % 10 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
    
            else:
                print('Unkown Status')
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindow()
                break





