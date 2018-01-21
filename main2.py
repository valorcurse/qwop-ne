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
    print("Creating NEAT object")
    
    qwop = QWOP()

    # cv2.imshow("image", qwop.runningTrack())
    # cv2.waitKey()

    keys = [Key.Q, Key.W, Key.O, Key.P]
    running = True
    gameStarted = False
    while (running):
        qwop.grabImage()

        if (not gameStarted and qwop.isAtIntro()):
            gameStarted = True
            qwop.startGame()
        
        if (gameStarted and not qwop.isAtIntro()):
            if (qwop.isPlayable()):
                key = random.choice(keys)
                # qwop.pressKey(key)
            else:
                running = False

        
        cv2.imshow("image", qwop.runningTrack())
        cv2.waitKey(1)

    qwop.stop()
    cv2.destroyAllWindows()