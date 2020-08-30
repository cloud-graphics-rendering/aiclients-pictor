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
        self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/newspace/AI-Bots-Client/training_scripts/AI-models/inmindvr/frozen_inference_graph.pb'
        self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/newspace/AI-Bots-Client/training_scripts/AI-models/inmindvr/label_map.pbtxt'
        self.NUM_CLASSES 	= 3
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

def nothing(x):
  pass

if __name__ == '__main__':
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    detector = TOD()
    detection_graph = detector._load_model()
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("redh",   "detection",0,255,nothing)
    cv2.createTrackbar("greenh", "detection",0,255,nothing)
    cv2.createTrackbar("blueh",  "detection",0,255,nothing)
    cv2.createTrackbar("redl",   "detection",0,255,nothing)
    cv2.createTrackbar("greenl", "detection",0,255,nothing)
    cv2.createTrackbar("bluel",  "detection",0,255,nothing)

    with mss.mss(display=':0.0') as sct:
        region={'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=config) as sess:
                while(True):
                    redh   = cv2.getTrackbarPos("redh",   "detection")
                    redl   = cv2.getTrackbarPos("redl",   "detection")
                    greenh = cv2.getTrackbarPos("greenh", "detection")
                    greenl = cv2.getTrackbarPos("greenl", "detection")
                    blueh  = cv2.getTrackbarPos("blueh",  "detection")
                    bluel  = cv2.getTrackbarPos("bluel",  "detection")
                    last_time = time.time()
                    screen = np.array(sct.grab(region))
                    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                    #lower = np.array([redl,greenl,bluel], dtype = "uint8")
                    #upper = np.array([redh,greenh,blueh], dtype = "uint8")
                    lower = np.array([171,0,21], dtype = "uint8")
                    upper = np.array([255,44,100], dtype = "uint8")
                    mask = cv2.inRange(screen, lower, upper)
                    output = cv2.bitwise_and(screen, screen, mask = mask)
                    img1gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
                    retval1, img1bin = cv2.threshold(img1gray,50,255,0)
                    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    e = cv2.erode(img1bin, el, iterations=1)
                    cv2.imshow("e", e)
                    d = cv2.dilate(e, el, iterations=20)
                    #result = cv2.bitwise_and(e, d)
                    cv2.imshow("d", d)
                    d,contours,hierarchy = cv2.findContours(d, 1, 2)
                    for contour in contours:
                        #(x,y),radius = cv2.minEnclosingCircle(contour)
                        #center = (int(x),int(y))
                        #radius = int(radius)
                        radius = 15
                        br = cv2.boundingRect(contour)
                        m = cv2.moments(contour)
                        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                        print("Center")
                        print(center)
                        cv2.circle(screen,center,radius,(0,255,0),2)
                    cv2.imshow("circle", screen)
                   #if len(contours) > 0:
                   #    cnt = contours[0]
                   #    (x,y),radius = cv2.minEnclosingCircle(cnt)
                   #    center = (int(x),int(y))
                   #    radius = int(radius)
                   #    cv2.circle(screen,center,radius,(0,255,0),2)
                   #    cv2.imshow("circle", screen)
                   #    print("find")
                   #else:
                   #    print("not find")
                   #    None
                    image = detector.detect(sess, detection_graph, screen)
                    print('Frame took {} seconds'.format(time.time()-last_time))
                    #cv2.imshow("detection", img1bin)
                    cv2.imshow("detection", np.hstack([img1gray,img1bin]))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindow()
                        break





