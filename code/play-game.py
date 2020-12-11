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
        """
        Handles the interaction with the game. Sets up and starts the game,
        gets realtime screenshots, feeds the image into the model, performs the
        action.
        """

        print("Welcome to TempleFlow!")

        self.action_dict = { # used to convert predictions to keystrokes
            "turn_left" : 'a',
            "turn_right" : 'd',
            "jmp" : 'w',
            "slide" : 's',
            "lean_left" : 'left',
            "lean_right" : 'right'
        }

        self.model = Model(False) # setup model and play
        self.game_running = True
        self.play()

    def play(self):
        """
        Performs the information described in init
        """

        print("Getting game region...")

        # locate the game on the screen for taking screenshots
        game_start_screen = '../data/screens/start_screen.png'
        game_region = pyautogui.locateOnScreen(game_start_screen, confidence=0.70)

        if game_region == None:
            print("Error getting game region...")
            return

        print("Starting game...")

        # locate and click play
        start_button = '../data/buttons/start_button.png'

        if self.comp_on_screen(start_button):
            self.click_button(start_button)
        else:
            print("Error getting start button...")
            return

        print("Ready to play!")
<<<<<<< HEAD

        # take screenshots and get predictions
        while self.game_running:
            screenshot = pyautogui.screenshot(region=game_region).resize((self.model.img_width, self.model.img_height))
            action = self.model(screenshot)

            self.perform_action(action)
=======
       # self.perform_action("jmp")
        
        while self.game_running:
            #ax`time.sleep(0.5);a
            # take screenshot


            if ready_to_predict():
                print("Input detected")
                t = time.time()
                screenshot = pyautogui.screenshot(region=game_region).resize((self.model.img_width, self.model.img_height))
                t1 = time.time() - t
                print(t1)
#                 check for end game button
#                 maybe do this in parallel?

#                end_game_image = '../data/buttons/end_run.png'
#
#                if self.comp_on_screen(end_game_image):
#                    print("Done playing!")
#                    self.game_running = False

           # time.sleep(self.delay)
            # feed screenshot into model
                t = time.time()
                action = self.model(screenshot)
                t1 = time.time() - t
                print(t1)
                self.perform_action(action)

                # feed output into perform_action
>>>>>>> 1777c3069e7c01e606837004f547a22affd6ed62

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
        """
        Purpose: click the given button
        Args: an of the button to click
        Return: none
        """
        x, y = pyautogui.locateCenterOnScreen(button_image, confidence=0.75) # get location of button
        x, y = x // 2, y // 2 # bug fix for mac

        clicks = 1
        if button_image == '../data/buttons/start_button.png': # make sure to select BlueStacks and then start
            clicks = 2

        pyautogui.click((x,y ), clicks=clicks)
        pyautogui.moveTo(x, y)

    def comp_on_screen(self, image):
        """
        Purpose: checks if the given image is on the screen
        Args: the item to look for
        Returns: True for on screen false otherwise
        """

        region = pyautogui.locateCenterOnScreen(image, confidence=0.75)
        
        if region == None:
            return False
        
        return True

if __name__ == '__main__':
    game = Game() # create and run a game