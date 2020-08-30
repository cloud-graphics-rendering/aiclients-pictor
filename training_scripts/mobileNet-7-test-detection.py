# -*- coding: utf-8 -*-
# test-detection.py

import numpy as np
import cv2
import time
import os
import mss
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TOD(object):
    def __init__(self):
        #self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/models-14/frozen_inference_graph.pb'
        #self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/workspace/bench-datasets/0ad/model/frozen_inference_graph.pb'
        #self.PATH_TO_CKPT 	= './AI-models/nasaweb/frozen_inference_graph.pb'
        self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/newspace/AI-Bots-Client/training_scripts/AI-models/redeclipse/frozen_inference_graph.pb'
        #self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/newspace/AI-Bots-Client/training_scripts/AI-models/inmind/frozen_inference_graph.pb'
        #self.PATH_TO_CKPT 	= '/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/training-model/frozen_inference_graph.pb'
        #self.PATH_TO_LABELS 	= '/media/lty/newspace/BenchmarkFrameWork/data-sets/nasaweb/label_map.pbtxt'
        self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/newspace/AI-Bots-Client/training_scripts/AI-models/redeclipse/label_map.pbtxt'
        #self.PATH_TO_LABELS 	= './AI-models/nasaweb/label_map5.pbtxt'
        #self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/workspace/bench-datasets/0ad/model/label_map18.pbtxt'
        #self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/label_map14.pbtxt'
        self.NUM_CLASSES 	= 6
        self.category_index 	= self._load_label_map()

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

    def detect(self, sess, detection_graph, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor	  = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes		  = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores		  = detection_graph.get_tensor_by_name('detection_scores:0')
        classes		  = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections	  = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
               [boxes, scores, classes, num_detections],
               feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
               image,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),\
               np.squeeze(scores),
               self.category_index,
               use_normalized_coordinates=True,
               line_thickness = 8)
        return image

if __name__ == '__main__':
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    detector = TOD()
    detection_graph = detector._load_model()

    with mss.mss(display=':0.0') as sct:
        region={'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=config) as sess:
            #with tf.Session(graph=detection_graph) as sess:
                while(True):
                    last_time = time.time()
                    screen = np.array(sct.grab(region))
                    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                    image = detector.detect(sess, detection_graph, screen)
                    print('Frame took {} seconds'.format(time.time()-last_time))

                    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                    cv2.imshow("detection", image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindow()
                        break





