# -*- coding: utf-8 -*-
# lstm_playTux.py

import numpy as np
import cv2
import time
import os
import tensorflow as tf
import mss
from TOD_OLD import TOD, collect_data

if __name__ == '__main__':
    
    file_name = '../training_data/raw-data/training_data' + str(int(time.time())) + '.npy'
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
        detector = TOD(sct)
        detection_graph = detector._load_model()
        start = 0

        with tf.Session(graph=detection_graph) as sess:
            while(True):
                print('start:',start)
                start = detector.gotoDestination(start,training_data, detection_graph, sess)

                if collect_data ==1 and len(training_data) % 10 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
    
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindow()
                    break





