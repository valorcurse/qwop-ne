from qwop import QWOP, Key
# from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

import torch
from torch.autograd import Variable

import multiprocessing
from multiprocessing import Pool, Queue, Value
import multiprocessing.managers as managers

from matplotlib import pyplot

import numpy as np
import cv2
import time

import os
import sys
import pickle
import random
# import _pickle as cPickle
import argparse

def profiler(phenotype):
    cProfile.runctx('testOrganism(phenotype)', globals(), locals(), 'prof.prof')

def testOrganism(phenotype, instances, finishedIndex, nrOfPhenotypes):
    running = True
    gameStarted = False

    fitnessScore = 0

    qwop = instances.get()
    if (phenotype.toDraw):
        qwop.grabImage()
        phenotype.draw(qwop.runningTrack())
        phenotype.toDraw = False

    startTime = None
    while (running):
        qwop.grabImage()
        
        if (not gameStarted):
            gameStarted = True
            qwop.startGame()
            # time.sleep(1)
        else:
            if (qwop.isPlayable()):
                previousFitnessScore = fitnessScore
                fitnessScore = qwop.score()

                if fitnessScore == previousFitnessScore:
                    if startTime == None:
                        startTime = time.time()
                    else:
                        # print("\rTime standing still: " + str(time.time() - startTime), end='')
                        if (time.time() - startTime) > 2.0:
                            running = False
                else:
                    startTime = None

                predicted = np.argmax(phenotype.update(qwop.runningTrack().flatten()), axis=0)
                qwop.pressKey(Key(predicted).name)
            else:
                running = False

    finishedIndex.value += 1
    print("\rFinished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")", end='')

    instances.put(qwop)

    # print("\rFinished phenotype ("+ i +"/"+ nrOfPhenotypes +")", end='')
    # print("Finished phenotype ("+ i +"/"+ nrOfPhenotypes +")")

    return fitnessScore

class QueueManager(managers.BaseManager):
    pass # Pass is really enough. Nothing needs to be done here.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("saveFileName")
    parser.add_argument("--load")

    args = parser.parse_args()

    saveFile = args.saveFileName
    saveDirectory = "saves"
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    sys.setrecursionlimit(10000)
    multiprocessing.set_start_method('spawn')
    QueueManager.register("QWOP", QWOP)
    queueManager = QueueManager()
    queueManager.start()

    nrOfInstances = 6
    nrOfOrgamisms = 6

    instances = multiprocessing.Manager().Queue()
    for i in range(nrOfInstances):
        newQWOP = queueManager.QWOP()
        print(newQWOP)
        instances.put(newQWOP)

    neat = None
    if (args.load):
        print("Loading NEAT object from file")
        with open(saveDirectory + "/" + args.load, "rb") as load:
            neat, innovations = pickle.load(load)
    else:
        print("Creating NEAT object")
        neat = NEAT(nrOfOrgamisms, 1054, 4)

    # pyplot.ion()
    # pyplot.show()
    pyplot.show(block=False)

    while True:
        results = {}

        randomPhenotype = random.choice(neat.phenotypes)
        randomPhenotype.toDraw = True

        pool = Pool(nrOfInstances)
        finishedIndex = multiprocessing.Manager().Value('i', 0)
        for i, phenotype in enumerate(neat.phenotypes):
            results[i] = pool.apply_async(testOrganism, (phenotype, instances, finishedIndex, len(neat.phenotypes)))
        pool.close()
        pool.join()

        fitnessScores = [result.get() for func, result in results.items()]

        # pyplot.pause(0.05)

        print("")
        print("-----------------------------------------------------")
        print(fitnessScores)
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print("Generation: " + str(neat.generation))
        print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        print("Number of genomes: " + str(len(neat.genomes)))
        print("Number of species: " + str(len(neat.species)))
        print("Number of phenotypes: " + str(len(neat.phenotypes)))
        
        with open(saveDirectory + "/" + saveFile, 'wb') as output:
            pickle.dump([neat, innovations], output, pickle.HIGHEST_PROTOCOL)