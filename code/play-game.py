import pyautogui
import time

# mappings from action name to corresponding key

action_dict = {
    "turn left" : 'a',
    "turn right" : 'd',
    "jump" : 'w',
    "duck" : 's',
    "lean left" : 'left',
    "lean right" : 'right'
}

def perform_action(action, duration=0):
    """

    This function takes in an action name and presses the coresponding key
    for the given duration.

    args: 
        action = the action name to take
        duration = how long to perform the action

    return: None
    """

    if action == None: # continue moving forward
        return

    key_to_press = action_dict[action] # get the key to press

    if duration == 0: # not leaning
        pyautogui.press(key_to_press) # press the key and done
        return

    start = time.time() 
    while time.time() - start < duration: # hold the key for the duration
        pyautogui.press(key_to_press)

def main():
    time.sleep(5) # delay to get set up
    perform_action("turn left", 5)

if __name__ == '__main__':
    main()