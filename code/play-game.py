import pyautogui
import time

from pynput import mouse

# mappings from action name to corresponding key

class Game():
    def __init__(self):

        self.action_dict = {
            "turn left" : 'a',
            "turn right" : 'd',
            "jump" : 'w',
            "duck" : 's',
            "lean left" : 'left',
            "lean right" : 'right'
        }

        self.clicks = 0

    def perform_action(self, action, duration=0):
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

        key_to_press = self.action_dict[action] # get the key to press

        if duration == 0: # not leaning
            pyautogui.press(key_to_press) # press the key and done
            return

        start = time.time() 
        while time.time() - start < duration: # hold the key for the duration
            pyautogui.press(key_to_press)



    def on_click(self, x, y, button, pressed):
        if pressed:
            print(x, y)
            self.clicks += 1

        if self.clicks == 4:
            return False

    def get_game_bounds(self):
        with mouse.Listener(on_click=self.on_click) as listener:
            try:
                listener.join()
            except:
                return

def main():
    # time.sleep(5) # delay to get set up
    # perform_action("turn left", 5)

    get_game_bounds()
    

if __name__ == '__main__':
    main()