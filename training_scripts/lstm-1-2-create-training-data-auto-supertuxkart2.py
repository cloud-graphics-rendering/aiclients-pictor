import numpy as np
import sys
import mss
import os
import cv2
import time
import tools
import keyboard_action
from pykeyboard import PyKeyboard
from datetime import datetime

def nothing(x):
        pass

def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]

    src = np.float32([[843, 518],    # br
                      [1067, 519],    # bl
                      [321, 900],   # tl
                      [1530, 900]])  # tr
    dst = np.float32([[843, 518],       # br
                      [1067, 519],       # bl
                      [843, 900],       # tl
                      [1067, 900]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def ProperBricksEdgePoint(box, brickLongEdgePoint):
    """
    Filter PROPER size rectangles and get the endPoint of longer edge.
    """
    dist1 = tools.pointsdist(box[0], box[1])
    dist2 = tools.pointsdist(box[1], box[2])
    if(dist1>dist2 and dist2>10 and dist1<400):
       brickLongEdgePoint.append([box[0],box[1]])
    elif(dist1<dist2 and dist1>10 and dist2 <400):
       brickLongEdgePoint.append([box[1],box[2]])
    else:
       pass
    return brickLongEdgePoint

def calculate_angle(endPoint):
    v0=np.array(endPoint[0])-np.array(endPoint[1])
    v1=np.array([0,1])-np.array([0,0])
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

if __name__ == '__main__':
    associate_flag 	= 0
    RUNNING_TIME	= 300
    AI_BOTS_DIR	= ""
    RESULT_DIR  	= ""
    BIND_CPU  	= 0
    HUMAN_RUN  	= 0
    Reso_Width      = 1920
    Reso_Hight      = 1080
    MultipleMode    = 1
    last_angle = 0
    training_data = []
    collect_data = 1
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
    terminalFocus=[200,200,200,200]
    supertuxkart_restart = [985,985,1030,1030]
    keyboard_action.mouse_click(terminalFocus)
    time.sleep(1)
    key_board = PyKeyboard()
    key_board.type_string('cd $CGR_BENCHMARK_PATH/')
    key_board.tap_key(key_board.enter_key)
    key_board.type_string('./collectData.sh supertuxkart '+str(associate_flag)+' '+str(RUNNING_TIME)+' '+str(BIND_CPU)+' '+str(HUMAN_RUN)+' '+str(MultipleMode)+' &')
    key_board.tap_key(key_board.enter_key)
    time.sleep(5)
    RUNNING_TIME -= 30

    FILE_NAME = RESULT_DIR+'/cv_action_time.csv'
    output_file = open(FILE_NAME,"w")
    columnTitleRow = "DATE TIME CV_TIME ACTION_TIME\n"
    output_file.write(columnTitleRow)

    file_name = '../training_data/supertuxkart2/raw-data/training_data' + str(int(time.time())) + '.npy'
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
        region={'top': 0, 'left': 0, 'width': Reso_Width, 'height': Reso_Hight}
        count = 0
        start_time = time.time()
        cur_time = time.time()   
        while((cur_time - start_time <= RUNNING_TIME) and (HUMAN_RUN==0)):
            cv_start = time.time()
            count+=1
            print(count)
            if(count >= 100):
                keyboard_action.rescue()
                keyboard_action.mouse_click(supertuxkart_restart)
                count = 0
            last_time = time.time()
            img = np.array(sct.grab(region))
            img_birdeye, M, Minv = birdeye(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), verbose=True)

            huh=95
            hul=64
            gray = cv2.cvtColor(img_birdeye, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,hul,huh,1)
            _,contours,h = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            brickLongEdgePoint=[]
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                if len(approx)==4:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    #cv2.drawContours(img_birdeye,[box],0,(0,0,255),2)
                    brickLongEdgePoint = ProperBricksEdgePoint(box, brickLongEdgePoint)
                else:
                    pass
            cv_end   = time.time()
            if(len(brickLongEdgePoint)==0):# refer the last one.
                keyboard_action.go_up1()
                pass
            else:
                angle = 0
                for points in brickLongEdgePoint:
                    angle += calculate_angle(points)
                angle = angle/len(brickLongEdgePoint)
                if(angle > 15):
                    training_data.append([[last_angle, angle],[1,0,0]])
                    keyboard_action.turn_left1()
                elif(angle < -15):
                    training_data.append([[last_angle, angle],[0,0,1]])
                    keyboard_action.turn_right1()
                else:
                    training_data.append([[last_angle, angle],[0,1,0]])
                    keyboard_action.go_up1()
                last_angle = angle
            if collect_data ==1 and len(training_data) % 10 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            cur_time = time.time()   
            row = str(datetime.now())+" "+str((cv_end-cv_start)*1000)+" "+str((cur_time-cv_end)*1000)+"\n"
            output_file.write(row)
        output_file.close()  

