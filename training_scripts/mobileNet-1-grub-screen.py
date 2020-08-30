# -*- coding: utf-8 -*-
# grub-screen.py

import cv2
import mss
import os
import sys
import time
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please input DIR to save screen shots in following format:")
        print("Format: python ./grub-screen.py /home/xxx/DIR/ ,eg: /home/tianyiliu/Documents/workspace/gaming/myprojects/figures/supertuxkart/rawfigures/train")
        print("Directory eg: /home/tianyiliu/Documents/workspace/gaming/myprojects/figures/supertuxkart/rawfigures/train")
        sys.exit()
    
    save_path = sys.argv[1]

    if (not os.path.exists(save_path)):
        print("directory does not exits!")
        sys.exit()

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    with mss.mss() as sct:
        region={'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        frame_num = 1
        while(True):
            screen    	= np.array(sct.grab(region))
            #time.sleep(0.1)
            #save_name 	= save_path + time.strftime("%b-%d-%Y-%H-%M-%S", time.gmtime())+'.jpg'
            save_name 	= save_path + datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.jpg'
            cv2.imwrite(save_name,screen) 
            frame_num=frame_num + 1;
            print('Frames grabed:', frame_num)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindow()
                break





