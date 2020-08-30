# -*- coding: utf-8 -*-
# test-detection.py

import numpy as np
import cv2
import time
import os
import sys
import mss
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/models-13/frozen_inference_graph.pb'
        self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/label_map.pbtxt'
        self.NUM_CLASSES 	= 13
        self.detection_graph 	= self._load_model()
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

                vis_util.visualize_boxes_and_labels_on_image_array(
                       image,
                       np.squeeze(boxes),
                       np.squeeze(classes).astype(np.int32),\
                       np.squeeze(scores),
                       self.category_index,
                       use_normalized_coordinates=True,
                       line_thickness = 8)

        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        print('Hello1, exit')
        cv2.imshow("detection", image)
        print('Hello2, exit')
        if cv2.waitKey(100) & 0xFF == ord('q'):
            sys.exit()
        print('Hello3, exit')

if __name__ == '__main__':
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    image = cv2.imread('/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/figures/supertuxkart/rawfigures/test/May-29-2018-19-01-34.jpg')
    detector = TOD()
    detector.detect(image)
    cv2.destroyAllWindows()
    print('Hello4, exit')



