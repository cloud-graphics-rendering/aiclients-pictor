import keyboard
import pyautogui
from pymouse import PyMouse
import time
import random
from datetime import datetime

def random_move():
    rand_num = random.randint(0,3)
    if rand_num == 0:
        keyboard.press('up')
        time.sleep(0.5)
        keyboard.release('up')
    elif rand_num == 1:
        keyboard.press('left')
        time.sleep(0.5)
        keyboard.release('left')
    elif rand_num == 2:
        keyboard.press('down')
        time.sleep(0.5)
        keyboard.release('down')
    else:
        keyboard.press('right')
        time.sleep(0.5)
        keyboard.release('right')

def moveTo(destx, desty, duration=1):
    x,y = pyautogui.position()
    pixelsx = destx-x
    pixelsy = desty-y
    moves = max(abs(pixelsx), abs(pixelsy), 1)
    
    avgpixelsx = float(pixelsx)/moves
    avgpixelsy = float(pixelsy)/moves
    dur = float(duration)/moves
    i = 1
    while i <= moves:
        offsetx = avgpixelsx * i
        offsety = avgpixelsy * i
        pyautogui.moveTo(x+offsetx, y + offsety, 0, False, False)
        time.sleep(dur)
        i=i+1

def mouse_click(region, button=1, rand=0):
    m=PyMouse()
    if rand == 0:
        m.click(int((region[0]+region[1])/2), int((region[2]+region[3])/2), button)
    else:
        click_x = random.randint(int(region[0]), int(region[1]))
        click_y = random.randint(int(region[2]), int(region[3]))
        m.click(click_x, click_y, button)

def mouse_double_click(region, button=1, rand=0):
    mouse_click(region, button, rand);
    mouse_click(region, button, rand);

def check_keys():
    if keyboard.is_pressed('left'):
        keys = [1,0,0]
    elif keyboard.is_pressed('right'):
        keys = [0,0,1]
    elif keyboard.is_pressed('up'):
        keys = [0,1,0]
    else:
        keys = [0,0,0]

    return keys

def check_4keys():
    if keyboard.is_pressed('left'):
        key = 1
    elif keyboard.is_pressed('right'):
        key = 2
    elif keyboard.is_pressed('up'):
        key = 3
    elif keyboard.is_pressed('down'):
        key = 4
    else:
        key = 0

    return key

def check_updown():
    if keyboard.is_pressed('up'):
        updown = [1,0,0]
    elif keyboard.is_pressed('down'):
        updown = [0,0,1]
    else:
        updown = [0,1,0]

    return updown

def check_leftright():
    if keyboard.is_pressed('left'):
        leftright = [1,0,0]
    elif keyboard.is_pressed('right'):
        leftright = [0,0,1]
    else:
        leftright = [0,1,0]

    return leftright

def stop_move():
    keyboard.release('up')
    keyboard.release('down')
    #pyautogui.keyUp('up')
    #pyautogui.keyUp('down')

def move_up():
    keyboard.release('down')
    keyboard.release('left')
    keyboard.release('right')
    keyboard.press('up')
    #pyautogui.keyDown('up')
    #pyautogui.keyUp('left')
    #pyautogui.keyUp('right')

def go_up():
    keyboard.release('down')
    keyboard.release('left')
    keyboard.release('right')
    keyboard.press('up')
    time.sleep(0.4)
    keyboard.release('up')
    #pyautogui.keyDown('up')
    #pyautogui.keyUp('left')
    #pyautogui.keyUp('right')

def go_up1():
    keyboard.release('down')
    keyboard.release('left')
    keyboard.release('right')
    keyboard.press('up')
    time.sleep(0.2)
    keyboard.release('up')

def rescue():
    keyboard.press('backspace')
    time.sleep(0.1)
    keyboard.release('backspace')

def tap_key(key, n=1):
    while n>0:
        keyboard.press(key)
        time.sleep(0.1)
        keyboard.release(key)
        n -= 1

def go_back1():
    keyboard.release('up')
    keyboard.release('left')
    keyboard.press('right')
    keyboard.press('down')
    time.sleep(0.2)
    keyboard.release('right')
    keyboard.release('down')

def go_back():
    keyboard.release('up')
    keyboard.release('left')
    keyboard.release('right')
    keyboard.press('down')
    time.sleep(0.4)
    keyboard.release('down')

def move_back():
    keyboard.release('up')
    keyboard.press('down')
    keyboard.press('left')
    time.sleep(0.7)
    keyboard.release('left')
    keyboard.press('up')
    time.sleep(1.2)
    #pyautogui.keyUp('up')
    #pyautogui.keyDown('down')

def turn_right():
    keyboard.release('left')
    keyboard.press('up')
    keyboard.press('right')
    time.sleep(0.4)
    keyboard.release('right')
    keyboard.release('up')

def turn_left():
    keyboard.release('right')
    keyboard.press('up')
    keyboard.press('left')
    time.sleep(0.4)
    keyboard.release('left')
    keyboard.release('up')

def turn_right1():
    keyboard.release('left')
    keyboard.press('up')
    keyboard.press('right')
    time.sleep(0.2)
    keyboard.release('right')
    keyboard.release('up')

def turn_left1():
    keyboard.release('right')
    keyboard.press('up')
    keyboard.press('left')
    time.sleep(0.2)
    keyboard.release('left')
    keyboard.release('up')

def quick_right():
    keyboard.release('left')
    keyboard.press('right')
    keyboard.press('up')
    #pyautogui.keyUp('left')
    #pyautogui.keyDown('right')

def quick_left():
    keyboard.release('right')
    keyboard.press('left')
    keyboard.press('up')
    #pyautogui.keyUp('right')
    #pyautogui.keyDown('left')

def turn_oneDirection(last_turn_left=0):
    if last_turn_left:
        turn_left()
        last_turn_left = 1
    else:
        turn_right()
        last_turn_left = 0
    return last_turn_left

def turn_opsitDirection(last_turn_left=0):
    if last_turn_left:
        turn_right()
        last_turn_left = 0
    else:
        turn_left()
        last_turn_left = 1
    return last_turn_left

def take_action(updown_index=0, leftright_index=0):
    if updown_index == 0 and leftright_index == 0:
        keyboard_action.turn_left()
    elif updown_index == 0 and leftright_index == 2:
        keyboard_action.turn_right()
    elif updown_index == 0:
        keyboard_action.move_up()
    elif updown_index == 2:
        keyboard_action.stop_move()
    else:
        print('other mode')

def go_forward():
    keyboard.release('down')
    keyboard.press('up')

def go_backward():
    keyboard.release('up')
    keyboard.press('down')

def go_right():
    keyboard.release('left')
    keyboard.press('right')

def go_left():
    keyboard.release('right')
    keyboard.press('left')

def go_attack(m):
    m.click(1,1)
