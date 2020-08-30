import numpy as np
import cv2
import mss
import time
import keyboard_action
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

collect_data = 0

class TOD(object):
    def __init__(self,sct, model_dir, label_dir, num_class, pic_region, stops, shape, lastPos):
        #self.PATH_TO_CKPT 	= './AI-models/supertuxkart/frozen_inference_graph.pb'
        #self.PATH_TO_LABELS 	= './AI-models/supertuxkart/label_map13.pbtxt'
        self.PATH_TO_CKPT 	= model_dir
        self.PATH_TO_LABELS 	= label_dir
        self.NUM_CLASSES 	= num_class
        self.category_index 	= self._load_label_map()
        self.shape  		= shape
        self.all_stops 		= stops
        self.pixel_error 	= 25

        self.stuck_count 	= 0
        self.stuck_threshold 	= 4

        self.good_angle 	= 40
        self.last_position	= lastPos
        self.dst 		= self.all_stops[1]
        self.region 		= pic_region
        #self.region 		= {'top': 50, 'left': 64, 'width': 1280, 'height': 960}
        self.im_width           = self.region['width']
        self.im_height          = self.region['height']
        self.sct 		= sct

    def check_stops(self,position):
        index=-1
        pixel_error = self.pixel_error
        for stop in self.all_stops:
            index = index+1
            if position[0] > stop[0]-pixel_error and position[0] < stop[0] + pixel_error and position[1] > stop[1] - pixel_error and position[1] < stop[1] + pixel_error:
                return index
        return 10        

    def samePositionAsBefore(self, cur_position=[1,1]):
        if cur_position[0] == 140 and cur_position[1] == 766:
            cur_position = self.last_position
            return False
        #print("stuck_count:",self.stuck_count)
        if abs(self.last_position[0]-cur_position[0])<1 and abs(self.last_position[1]-cur_position[1])<1:
            return True
        else:
            self.last_position = cur_position
            return False

    def samePosition(self, cur_position=[1,1]):
        #print("stuck_count:",self.stuck_count)
        if abs(self.last_position[0]-cur_position[0])<1 and abs(self.last_position[1]-cur_position[1])<1:
            self.stuck_count +=1
        else:
            self.stuck_count = 0
        self.last_position = cur_position

        if(self.stuck_count > self.stuck_threshold):
            self.stuck_count = 0
            return True
        else:
            return False

    def calculate_angle(self, cur=[0,1]):
        last= self.last_position
        dst = self.dst
        v0=np.array(cur)-np.array(last)
        v1=np.array(dst)-np.array(last)
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        return np.degrees(angle)

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

    def generate_feature_objects(self, start , detection_graph, sess):
        screen = np.array(self.sct.grab(self.region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        figure,obj_positions,image = self.detect_objects(screen,detection_graph, sess)
        angle  = self.calculate_angle(cur = figure[-2:])
        if start + 1 == 8:
            target_num = 1
        else:
            target_num = start+1

        if(self.check_stops(figure[-2:]) != target_num):
            feature_vector = [[self.dst + self.last_position + figure[-2:]],[angle]]
        else:
            start = start+1
            if start == 7:
                self.dst = self.all_stops[1]
            elif start ==8:
                start = 1
                self.dst = self.all_stops[start+1]
            else:
                self.dst = self.all_stops[start+1]
            feature_vector = [[self.dst + self.last_position + figure[-2:]],[angle]]
        return start, feature_vector, figure[-2:], obj_positions, image

    def generate_feature(self, start, detection_graph, sess):
        screen = np.array(self.sct.grab(self.region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        figure,image = self.detect(screen, detection_graph, sess)
        angle  = self.calculate_angle(cur = figure[-2:])
        if start + 1 == 8:
            target_num = 1
        else:
            target_num = start+1

        if(self.check_stops(figure[-2:]) != target_num):
            feature_vector = [[self.dst + self.last_position + figure[-2:]],[angle]]
        else:
            start = start+1
            if start == 7:
                self.dst = self.all_stops[1]
            elif start ==8:
                start = 1
                self.dst = self.all_stops[start+1]
            else:
                self.dst = self.all_stops[start+1]
            feature_vector = [[self.dst + self.last_position + figure[-2:]],[angle]]
        return start, feature_vector, figure[-2:], image

    def gotoDestination(self, start, training_data, detection_graph, sess):
        screen = np.array(self.sct.grab(self.region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        figure,_ = self.detect(screen,detection_graph, sess)
        angle  = self.calculate_angle(cur = figure[-2:])
        if start + 1 == 8:
            target_num = 1
        else:
            target_num = start+1
        
        if(self.check_stops(figure[-2:]) != target_num):
            while(abs(angle) > self.good_angle):
                if angle > self.good_angle:
                    if self.samePosition(cur_position = figure[-2:]):
                        if collect_data == 1:
                            #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,0,1]])
                            training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,0,1]])
                        keyboard_action.move_back()
                        return start
                    if collect_data == 1:
                        #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,0,1]])
                        training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,0,1]])
                    keyboard_action.turn_right()
                elif angle < -self.good_angle:
                    if self.samePosition(cur_position = figure[-2:]):
                        if collect_data == 1:
                            #print([[self.dst + self.last_position + figure[-2:]],[angle],[1,0,0]])
                            training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[1,0,0]])
                            keyboard_action.move_back()
                        return start
                    if collect_data == 1:
                        #print([[self.dst + self.last_position + figure[-2:]],[angle],[1,0,0]])
                        training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[1,0,0]])
                    keyboard_action.turn_left()
                else:
                    return start
                self.last_position = figure[-2:]
                screen = np.array(self.sct.grab(self.region))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                figure,_ = self.detect(screen,detection_graph, sess)
                angle = self.calculate_angle(cur = figure[-2:])
        
            if self.samePosition(cur_position = figure[-2:]):
                if collect_data == 1:
                    #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,1,0]])
                    training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,1,0]])
                keyboard_action.move_back()
                return start
            if collect_data == 1:
                #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,1,0]])
                training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,1,0]])
            keyboard_action.move_up()
        else:
            start = start+1
            if start == 7:
                self.dst = self.all_stops[1]
            elif start ==8:
                start = 1
                self.dst = self.all_stops[start+1]
            else:
                self.dst = self.all_stops[start+1]
                
            if angle > self.good_angle:
                if collect_data == 1:
                    #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,0,1]])
                    training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,0,1]])
                keyboard_action.turn_right()
            elif angle < -self.good_angle:
                if collect_data == 1:
                    #print([[self.dst + self.last_position + figure[-2:]],[angle],[1,0,0]])
                    training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[1,0,0]])
                keyboard_action.turn_left()
            else:
                if collect_data == 1:
                    #print([[self.dst + self.last_position + figure[-2:]],[angle],[0,1,0]])
                    training_data.append([[self.dst + self.last_position + figure[-2:]],[angle],[self.shape+self.last_position+figure[-2:]],[0,1,0]])
                keyboard_action.move_up()
        self.last_position = figure[-2:]        
        return start

    def detect_objects(self, image, detection_graph, sess):
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor	  = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes		  = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores		  = detection_graph.get_tensor_by_name('detection_scores:0')
        classes		  = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections	  = detection_graph.get_tensor_by_name('num_detections:0')
        

        (boxes, scores, classes, num_detections) = sess.run(
               [boxes, scores, classes, num_detections],
               feed_dict={image_tensor: image_np_expanded})
        im_width = self.im_width
        im_height = self.im_height
        i_num = 0
        for score in scores[0]:
            if score > 0.5:
                i_num+=1
        valid_boxes = boxes[0][0:i_num]
        valid_classes = list(classes[0])
        obj_classes = valid_classes[0:i_num] 
        obj_positions = []
        for box in valid_boxes:
            ymin, xmin, ymax, xmax = box
            [left, right, top, bottom] = [xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height]
            obj_positions.append([left,right,top,bottom])
        #obj_positions, obj_classes= vis_util.visualize_valid_boxes_and_labels_on_image_array(
        vis_util.visualize_boxes_and_labels_on_image_array(
               image,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),\
               np.squeeze(scores),
               self.category_index,
               use_normalized_coordinates=True,
               line_thickness = 8)
        #find rider and put it at head of list
        try:
            rider_index = obj_classes.index(2)
            del obj_classes[rider_index]
            obj_classes.insert(0,2)
            rider_position = obj_positions[rider_index]
            del obj_positions[rider_index]
            obj_positions.insert(0,rider_position)
        except:
            #print('RIDER is missing')
            rider_position = [910,1010,370,590]
            #rider_position = [560,660,360,580]
            obj_classes.insert(0,-1)
            obj_positions.insert(0,rider_position)
             
        #find profile and put it at head of list
        try:
            profile_index = obj_classes.index(1)
            del obj_classes[profile_index]
            obj_classes.insert(0,1)
            profile_position = obj_positions[profile_index]
            del obj_positions[profile_index]
            obj_positions.insert(0,profile_position)
        except:
            #print('PROFILE is missing')
            #profile_position = [345,355,495,505]
            profile_position = [135,145,761,771]
            obj_classes.insert(0,-1)
            obj_positions.insert(0,profile_position)

        feature_vector = [(obj_positions[0][0]+obj_positions[0][1])/2,
                          (obj_positions[0][2]+obj_positions[0][3])/2]
        if obj_classes[0] == -1:
            feature_vector = self.last_position
        return feature_vector, obj_positions, image

    def detect(self, image, detection_graph, sess):
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor	  = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes		  = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores		  = detection_graph.get_tensor_by_name('detection_scores:0')
        classes		  = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections	  = detection_graph.get_tensor_by_name('num_detections:0')
        
        time1 = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
               [boxes, scores, classes, num_detections],
               feed_dict={image_tensor: image_np_expanded})
        time2 = time.time()
        box_centers=vis_util.visualize_realboxes_and_labels_on_image_array(
               image,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),\
               np.squeeze(scores),
               self.category_index,
               use_normalized_coordinates=True,
               line_thickness = 8)
        time3 = time.time()
        if len(box_centers) > 0:
            feature_vector = box_centers[0]
        else:
            #feature_vector = [350,500]
            feature_vector = self.last_position
        return feature_vector, image
