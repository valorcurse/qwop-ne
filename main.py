from qwop import QWOP, Key
# from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

import torch
from torch.autograd import Variable

# import multiprocessing
# from multiprocessing import Pool, Queue, Value
# import multiprocessing.managers as managers

import multiprocess as multiprocessing
from multiprocess.pool import Pool
from multiprocess import Queue, Value
import multiprocess.managers as managers

from matplotlib import pyplot

import numpy as np
import cv2
import time

import os
import sys
# import pickle
import _pickle as pickle
# import dill
import random
import cProfile
import argparse
import traceback

# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

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

    differentKeysPressed = []
    startTime = None
    while (running):
        # qwop.grabImage()

        cv2.imshow("image", qwop.runningTrack())
        cv2.waitKey(1)
        
        if (not gameStarted):
            gameStarted = True
            qwop.startGame()
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

                inputs = qwop.runningTrack().flatten()
                outputs = phenotype.update(inputs)
                maxOutput = np.argmax(outputs, axis=0)
                predicted = Key(maxOutput)
                # qwop.pressKey(predicted)
                qwop.holdKey(predicted)

                if (not predicted in differentKeysPressed):
                    differentKeysPressed.append(predicted)
            else:
                running = False

    finishedIndex.value += 1
    print("\rFinished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")", end='')
    # print("Finished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")")

    instances.put(qwop)

    fitnessScore = max(0, fitnessScore)
    fitnessScore = pow(fitnessScore, len(differentKeysPressed))

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
    # multiprocessing.set_start_method('spawn')
    # multiprocessing.log_to_stderr()
    QueueManager.register("QWOP", QWOP)
    queueManager = QueueManager()
    queueManager.start()

    nrOfInstances = 4
    nrOfOrgamisms = 150

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

    pyplot.show(block=False)
    
    while True:
        results = {}

        # randomPhenotype = random.choice(neat.phenotypes)
        # randomPhenotype.toDraw = True

        pool = Pool(nrOfInstances)
        finishedIndex = multiprocessing.Manager().Value('i', 0)
        for i, phenotype in enumerate(neat.phenotypes):
            # results[i] = pool.apply_async(testOrganism, (phenotype, instances, finishedIndex, len(neat.phenotypes)))
            results[i] = pool.apply_async(LogExceptions(testOrganism), (phenotype, instances, finishedIndex, len(neat.phenotypes)))
        pool.close()
        pool.join()

        fitnessScores = [result.get() for func, result in results.items()]

        # testOrganism(neat.phenotypes[0], instances, finishedIndex, len(neat.phenotypes))
        # fitnessScores = [0] * len(neat.phenotypes)

        print("")
        print("-----------------------------------------------------")
        print(fitnessScores)
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print("Generation: " + str(neat.generation))
        # print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        # print("Number of genomes: " + str(len(neat.genomes)))
        print("Number of species: " + str(len(neat.species)))
        print("ID", "\t", "age", "\t", "fitness", "\t", "adj. fitness", "\t", "stag")
        for s in neat.species:
            print(s.ID, "\t", s.age, "\t", "{:1.4f}".format(max([m.fitness for m in s.members])), 
                "\t", "{:1.4f}".format(s.adjustedFitness), "\t", s.generationsWithoutImprovement)
        
    # with open(saveDirectory + "/" + saveFile, 'wb') as output:
        # pickle.dump([neat, innovations], output)