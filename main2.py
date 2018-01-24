from qwop import QWOP, Key
from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

import torch
from torch.autograd import Variable

import numpy as np
import cv2
import time
import random
from random import randint

if __name__ == '__main__':
    # print(cv2.getBuildInformation())
    
    qwop = QWOP()

    # cv2.imshow("image", qwop.runningTrack())
    # cv2.waitKey()

    previousFitnessScore = 0
    fitnessScore = 0
    startTime = None

    keys = [Key.Q, Key.W, Key.O, Key.P]
    running = True
    gameStarted = False
    while (running):

        if (not gameStarted and qwop.isAtIntro()):
            gameStarted = True
            qwop.startGame()
        
        if (gameStarted and qwop.isAtGameLost()):
            gameStarted = True
            qwop.startGame()

        if (gameStarted and not qwop.isAtIntro()):
            if (qwop.isPlayable()):
                # key = random.choice(keys)
                key = Key.W
                qwop.holdKey(key)
            # else:
                # gameStarted = False
                # running = False
            previousFitnessScore = fitnessScore
            fitnessScore = qwop.score()

            if fitnessScore == previousFitnessScore:
                if startTime == None:
                    startTime = time.time()
                else:
                    # print("\rTime standing still: " + str(time.time() - startTime), end='')
                    if (time.time() - startTime) > 2.0:
                        running = False
                        # qwop.pressKey(Key.R)
                        # qwop.startGame()
                        startTime = None

        
        # print(qwop.score())
        cv2.imshow("image", qwop.grayImage)
        cv2.waitKey(1)

    qwop.stop()
    cv2.destroyAllWindows()