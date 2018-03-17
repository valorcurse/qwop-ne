from qwop import QWOP, Key
from neat import NEAT
from genes import NeuronType

from genes import innovations

global innovations

# import torch
# from torch.autograd import Variable

import multiprocess as multiprocessing
from multiprocess.pool import Pool
from multiprocess import Queue, Value
import multiprocess.managers as managers

from matplotlib import pyplot

from prettytable import PrettyTable

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

windowPositions = [(0, 0), (450, 0), (900, 0), (0, 400), (450, 400)]

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

    displayStream = True

    qwop = instances.get()
    if (phenotype.toDraw):
        # qwop.grabImage()
        phenotype.draw(qwop.runningTrack())
        phenotype.toDraw = False

    if displayStream:
        cv2.namedWindow(str(qwop.getNumber()))
        pos = windowPositions[qwop.getNumber()]
        cv2.moveWindow(str(qwop.getNumber()), pos[0], pos[1])

    differentKeysPressed = []
    startTime = None
    while (running):
        # qwop.takeScreenshot()

        if displayStream:
            # qwop.showStream()
            cv2.imshow(str(qwop.getNumber()), qwop.getImage())
            cv2.waitKey(20)
        
        if (not gameStarted):
            gameStarted = True
            qwop.startGame()
        else:
            if qwop.isScoreSimilar():
                if startTime == None:
                    startTime = time.time()
                else:
                    # print("\rTime standing still: " + str(time.time() - startTime), end='')
                    if (time.time() - startTime) > 3.0:
                        fitnessScore = qwop.score()
                        running = False
            else:
                startTime = None

            inputs = qwop.runningTrack().flatten() / 255
            outputs = phenotype.update(inputs)
            maxOutput = np.argmax(outputs, axis=0)
            predicted = Key(maxOutput)
            qwop.holdKey(predicted)

            if (not predicted in differentKeysPressed):
                differentKeysPressed.append(predicted)

    finishedIndex.value += 1
    print("\rFinished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")", end='')
    # print("Finished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")")

    if displayStream:
        cv2.destroyWindow(str(qwop.getNumber()))
        
    instances.put(qwop)

    # phenotype.genome.distance = fitnessScore
    # phenotype.genome.uniqueKeysPressed = differentKeysPressed

    # fitnessScore = max(0, fitnessScore)
    distance = fitnessScore / 100.0
    if (fitnessScore > 0):
        fitnessScore = pow(fitnessScore, len(differentKeysPressed))

    return (distance, fitnessScore)

class QueueManager(managers.BaseManager):
    pass # Pass is really enough. Nothing needs to be done here.

if __name__ == '__main__':
    import pystuck; pystuck.run_server()

    parser = argparse.ArgumentParser()
    parser.add_argument("saveFolder")
    parser.add_argument("--load")

    args = parser.parse_args()

    saveDirectory = "saves/" + args.saveFolder
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    sys.setrecursionlimit(10000)
    # multiprocessing.set_start_method('spawn')
    multiprocessing.log_to_stderr()
    QueueManager.register("QWOP", QWOP)
    queueManager = QueueManager()
    queueManager.start()

    nrOfInstances = 1
    nrOfOrgamisms = 4

    instances = multiprocessing.Manager().Queue()
    for i in range(nrOfInstances):
        newQWOP = queueManager.QWOP(i)
        while (not newQWOP.isAtIntro()):
            # newQWOP.takeScreenshot()
            pass

        instances.put(newQWOP)

    neat = None
    if (args.load):
        print("Loading NEAT object from file")
        with open(saveDirectory + "/" + args.load, "rb") as load:
            neat, innovations = pickle.load(load)
    else:
        print("Creating NEAT object")
        # neat = NEAT(nrOfOrgamisms, 1054, 4)
        neat = NEAT(nrOfOrgamisms, 1890, 4)

    pyplot.show(block=False)
    
    stillRunning = True
    while stillRunning:
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

        distances = [result.get()[0] for func, result in results.items()]
        fitnessScores = [result.get()[1] for func, result in results.items()]

        # for i, phenotype in enumerate(neat.phenotypes):
            # fitnessScores.append(testOrganism(neat.phenotypes[0], instances, finishedIndex, len(neat.phenotypes)))
        # fitnessScores = [0] * len(neat.phenotypes)

        for p, d in zip(neat.phenotypes, distances):
            p.genome.distance = d

        print("")
        print("-----------------------------------------------------")
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print(distances)
        print("Generation: " + str(neat.generation))
        # print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        # print("Number of genomes: " + str(len(neat.genomes)))
        print("Number of species: " + str(len(neat.species)))
        table = PrettyTable(["ID", "age", "members", "max fitness", "adj. fitness", "avg. distance", "stag", "neurons", "links", "to spawn"])
        for s in neat.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                int(max([m.fitness for m in s.members])),                   # Max fitness
                "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
                # "{}".format(max([m.uniqueKeysPressed for m in s.members])), # Average unique keys
                s.generationsWithoutImprovement,                            # Stagnation
                int(np.mean([len(m.neurons)-1894 for m in s.members])),     # Neurons
                np.mean([len(m.links) for m in s.members]),                 # Links
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        # Save current generation
        saveFileName = args.saveFolder + "." + str(neat.generation)
        with open(saveDirectory + "/" + saveFileName, 'wb') as file:
            pickle.dump([neat, innovations], file)

        # Append to summary file
        with open(saveDirectory + "/summary.txt", 'a') as file:
            file.write("Generation " + str(neat.generation) + "\n")
            file.write(table.get_string())
            file.write("\n\n")

