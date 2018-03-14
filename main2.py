from qwop import QWOP, Key
# from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

# import torch
# from torch.autograd import Variable

import numpy as np
import cv2
import time
import random
from random import randint

from matplotlib import pyplot as plt

if __name__ == '__main__':
    # print(cv2.getBuildInformation())
    
    qwop = QWOP(0)

    # cv2.imshow("image", qwop.runningTrack())
    # cv2.waitKey()

    fitnessScore = 0

    while (not qwop.isAtIntro()):
        # print(qwop.takeScreenshot())
        # plt.imshow(qwop.image)
        # plt.show()
        pass

    keys = [Key.Q, Key.W, Key.O, Key.P]
    for i in range(25):
        running = True
        gameStarted = False
        startTime = None
        
        print("\rRun nr: " + str(i), end='')

        while (running):           

            if (not gameStarted):
                # if (qwop.isAtIntro()):
                gameStarted = True
                qwop.startGame()
            else:
                if qwop.isImageSimilar():
                    if startTime == None:
                        startTime = time.time()
                    else:
                        # print("\rTime standing still: " + str(time.time() - startTime), end='')
                        if (time.time() - startTime) > 2.0:
                            fitnessScore = qwop.score()
                            running = False
                else:
                    startTime = None

                key = random.choice(keys)
                # key = Key.W
                qwop.holdKey(key)
            
            # print(qwop.score())
            # cv2.imshow("image", qwop.grayImage)
            # cv2.imshow("image", qwop.runningTrack())
            # cv2.waitKey(1)

    qwop.stop()
    cv2.destroyAllWindows()