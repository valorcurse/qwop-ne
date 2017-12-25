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
from random import randint

if __name__ == '__main__':
    print("Creating NEAT object")
    
    qwop = QWOP()
    qwop.grabImage()

    cv2.imshow("image", qwop.runningTrack())
    cv2.waitKey()
    qwop.stop()