import pyautogui
import time

from pynput import mouse

# mappings from action name to corresponding key

class Game():
    def __init__(self, model):

        self.action_dict = {
            "turn left" : 'a',
            "turn right" : 'd',
            "jump" : 'w',
            "duck" : 's',
            "lean left" : 'left',
            "lean right" : 'right'
        }

        self.model = model
        self.game_running = False

    def play(self):

        # locate the game on the screen for taking screenshots
        game_start_screen = '../images/start_screen.png'
        game_region = pyautogui.locateOnScreen(game_start_screen)

        if game_region == None:
            print("Error getting game region...")
            return

        # locate and click play (maybe die to start to avoid different starts)
        play_game_image = '../images/play_game_button.png'
        play_game_region = pyautogui.locateOnScreen(play_game_image)

        if play_game_region == None:
            print("Error getting play game button region...")
            return

        pyautogui.click(play_game_region)

        # delay to start running
        time.sleep(5)

        while self.game_running:
            # take screenshot
            screenshot = pyautogui.screenshot(game_region)

            # check for end game button
            # maybe do this in parallel?
            
            end_game_image = '../images/end_run_button'
            end_game_region = pyautogui.locateOnScreen(end_game_image)

            if end_game_region != None:
                self.game_running = False

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

def click_button(button_image):
    button_region = pyautogui.locateOnScreen(button_image)
    print(button_region)
    pyautogui.click(button_region)


def main():
    time.sleep(5) # delay to get set up
    # perform_action("turn left", 5)

    end_game_image = '../images/end_run_button'
    click_button('brown logo.png')
    

if __name__ == '__main__':
    main()