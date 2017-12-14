from qwop import QWOP, Key
# from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

import torch
from torch.autograd import Variable

import multiprocessing
from multiprocessing import Pool
import multiprocessing.managers as managers
from multiprocessing import Queue
from multiprocessing import SimpleQueue

# import queue

import numpy as np
import cv2
import time

import os
import cProfile
import pickle

def profiler(phenotype):
    cProfile.runctx('testOrganism(phenotype)', globals(), locals(), 'prof.prof')

def testOrganism(phenotype, instances):
    running = True
    gameStarted = False

    fitnessScore = 0

    qwop = instances.get()

    while (running):
        qwop.grabImage()
        
        if (not gameStarted):
            gameStarted = True
            qwop.startGame()
        else:
            previousFitnessScore = fitnessScore
            fitnessScore = qwop.score()

            if fitnessScore == previousFitnessScore:
                if startTime == None:
                    startTime = time.time()
                else:
                    # print("\rTime standing still: " + str(time.time() - startTime), end='')
                    if (time.time() - startTime) > 3.0:
                        running = False
            else:
                startTime = None

            predicted = np.argmax(phenotype.update(qwop.runningTrack().flatten()), axis=0)
            qwop.pressKey(Key(predicted).name)

    instances.put(qwop)

    return fitnessScore

class QueueManager(managers.BaseManager):
    pass # Pass is really enough. Nothing needs to be done here.

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')
    QueueManager.register("QWOP", QWOP)
    queueManager = QueueManager()
    queueManager.start()
    # qwop = QWOP()

    # qwop.grabImage()

    # qwop.stop()

    nrOfInstances = 1
    nrOfOrgamisms = 20

    # qwop=QWOP()
    # print(test_pickle(qwop))

    # instances = SimpleQueue()
    instances = multiprocessing.Manager().Queue()
    for i in range(nrOfInstances):
        newQWOP = queueManager.QWOP()
        print(newQWOP)
        instances.put(newQWOP)

    print("Creating NEAT object")
    # instance = instances.get()
    # instance.grabImage()
    # neat = NEAT(nrOfOrgamisms, instance.runningTrack().size, 4)
    neat = NEAT(nrOfOrgamisms, 1840, 4)
    # instances.put(instance)

    while True:
        results = {}

        pool = Pool(nrOfInstances)
        for i, phenotype in enumerate(neat.phenotypes):
            results[i] = pool.apply_async(testOrganism, (phenotype, instances))
        pool.close()
        pool.join()

        fitnessScores = [result.get() for func, result in results.items()]

        print("-----------------------------------------------------")
        print(fitnessScores)
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print("Generation: " + str(neat.generation))
        print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        print("Number of genomes: " + str(len(neat.genomes)))
        print("Number of species: " + str(len(neat.species)))
        print("Number of phenotypes: " + str(len(neat.phenotypes)))
