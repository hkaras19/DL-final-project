import pyautogui
import time
from model import Model
import datetime
from PIL import Image
from pynput.keyboard import Listener, Key
import keyboard
# mappings from action name to corresponding key

class Game():
    def __init__(self):

        print("Welcome to TempleFlow!")

        self.action_dict = {
            "turn_left" : 'a',
            "turn_right" : 'd',
            "jmp" : 'w',
            "slide" : 's',
            "lean_left" : 'left',
            "lean_right" : 'right'
        }

        self.model = Model(False)
        self.game_running = True
        self.delay = 10 / 1000
        self.play()

    def play(self):
        print("Getting game region...")
        # locate the game on the screen for taking screenshots
        game_start_screen = '../data/screens/start_screen.png'
        game_region = pyautogui.locateOnScreen(game_start_screen, confidence=0.80)

        if game_region == None:
            print("Error getting game region...")
            return

        print("Starting game...")

        # locate and click play (maybe die to start to avoid different starts)
        start_button = '../data/buttons/start_button.png'

        if self.comp_on_screen(start_button):
            self.click_button(start_button)
        else:
            print("Error getting start button...")
            return

        # delay to start running
        #Print("Waiting...")
        # time.sleep(3.2)
               
        # time.sleep(6)
        print("Ready to play!")
        self.perform_action("jmp")
        
        # time.sleep(2.7)

        while self.game_running:
            # take screenshot
            screenshot = pyautogui.screenshot(region=game_region).resize((self.model.img_width, self.model.img_height))

         #   if ready_to_predict():
                # check for end game button
                # maybe do this in parallel?
#
#                end_game_image = '../data/buttons/end_run.png'
#
#                if self.comp_on_screen(end_game_image):
#                    print("Done playing!")
#                    self.game_running = False

            # time.sleep(self.delay)
            print("Input detected")
            # feed screenshot into model
            action = self.model(screenshot)
            # feed output into perform_action
            self.perform_action(action)

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

    def click_button(self, button_image):
        x, y = pyautogui.locateCenterOnScreen(button_image, confidence=0.75)
        x, y = x // 2, y // 2

        clicks = 1
        if button_image == '../data/buttons/start_button.png':
            clicks = 2

        pyautogui.click((x,y ), clicks=clicks)
        pyautogui.moveTo(x, y)

    def comp_on_screen(self, image):
        region = pyautogui.locateCenterOnScreen(image, confidence=0.75)
        
        if region == None:
            return False
        
        return True

def ready_to_predict():
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('p'):  # if key 'q' is pressed 
            return True
    except:
        return False

    return False

if __name__ == '__main__':
    # with Listener(on_press=on_press) as listener:
    #     listener.daemon = True
    #     listener.start()

    game = Game()

        
    
    
