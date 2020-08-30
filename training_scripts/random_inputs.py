import random
import time
import keyboard_action
import sys
from pykeyboard import PyKeyboard

start_time = time.time()
cur_time = start_time
time.sleep(10)
RUNNING_TIME = 240
if(len(sys.argv) > 1):
    RUNNING_TIME    = int(sys.argv[1])

key_board = PyKeyboard()
key_table = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','x','y','z','Up','Down','Left','Right']

keyboard_action.mouse_click([100,1800,100,800],random.randrange(1,3),1)

while (cur_time - start_time <= RUNNING_TIME):
    time.sleep(random.random())
    device = random.randrange(0,2)
    if device == 0: #keyboard
        keysymbol = key_table[random.randrange(0,len(key_table))]
        key_board.press_key(keysymbol)
        time.sleep(random.random())
        key_board.release_key(keysymbol)
    else:#Mouse
        keyboard_action.mouse_click([100,1800,100,800],random.randrange(1,3),1)
    cur_time = time.time()
