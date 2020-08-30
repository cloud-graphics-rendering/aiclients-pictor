# -*- coding: utf-8 -*-
# test-detection.py
import numpy as np
import time
import os
import mss
import keyboard_action

if __name__ == '__main__':
    file_name = '../training_data/redeclipse/training_data' + str(int(time.time())) + '.npy'
    training_data = []

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    output_keysList=[]
    while(True):
        time.sleep(0.2)
        output_keys = keyboard_action.check_4keys()
        output_keysList.insert(0,output_keys)
        if len(output_keysList) == 21:
            if output_keysList[20] == 0:
                figure_vector 	= [output_keysList[0:20],[1,0,0,0,0]]
                print(figure_vector)
                training_data.append(figure_vector)
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            elif output_keysList[20] == 1:
                figure_vector 	= [output_keysList[0:20],[0,1,0,0,0]]
                print(figure_vector)
                training_data.append(figure_vector)
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            elif output_keysList[20] == 2:
                figure_vector 	= [output_keysList[0:20],[0,0,1,0,0]]
                print(figure_vector)
                training_data.append(figure_vector)
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            elif output_keysList[20] == 3:
                figure_vector 	= [output_keysList[0:20],[0,0,0,1,0]]
                print(figure_vector)
                training_data.append(figure_vector)
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            elif output_keysList[20] == 4:
                figure_vector 	= [output_keysList[0:20],[0,0,0,0,1]]
                print(figure_vector)
                training_data.append(figure_vector)
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            else:
                None
            output_keysList.pop()

