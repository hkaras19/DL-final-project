# TempleFlow: Deep Learning Final Project
### Hunter Karas and Arun Kavishwar

## Final Writeup
https://docs.google.com/document/d/1LjoRqpEnWkFB6FJvjET3JLb6bRu2LMrfPzNkp3gJuiE/edit#

## Project Description
For the final project of csci1470 (Deep Learning), our team chose option two from the assignment handout and attempted to solve a new problem using deep learning. More specifically, we decided to create a deep learning program which learns how to play a “never-before-attempted” game. 

Our game of choice was the popular, mobile game Temple Run 2 originally created by Imangi Studios. Our AI player uses a transfer learned convolutional neural network (CNN) to classify real time screenshots of the game into six categories: turn_left, turn_right, jump, slide, lean_left, and lean_right. Our player then takes the predicted action and uses a python library, called pyautogui, to run the corresponding command in our Android game emulator (BlueStacks).

We trained our model by collecting and labeling around 13,000 images (over multiple training trials) and were able to get a test accuracy greater than 98%. Additionally, our model can correctly predict obstacles at runtime with some limitations (see results and challenges section).

At this final deadline, we can declare success on our target goal (see reflection). While we did not reach our stretch goal, of the AI player being able to navigate the game indefinitely, we were able to satisfy two of our other goals: our base goal and target goal. We will now describe the details of our project.

## How To Use
1. Download BlueStacks and install Temple Run 2 onto the emulator

2. Open a terminal window and place it next to the temple run game

Note: make sure you can see the whole game screen and that you start on the home screen (take the idol).
Note: you will need to install a few dependencies to run the project (do some pip installs and googling as the errors arise)

3. Run python3 play-game.py

4. Stop the predictions by stopping your terminal process
