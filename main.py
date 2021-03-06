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
import _pickle as pickle
import random
import cProfile
import argparse
import traceback

import stackimpact

windowPositions = [(0, 0), (450, 0), (900, 0), (0, 400), (450, 400)]

def testOrganism(phenotype, qwop):
    running = True
    gameStarted = False

    fitnessScore = 0

    displayStream = True
    windowName = str(phenotype.genome.ID)

    if (phenotype.toDraw):
        # qwop.grabImage()
        phenotype.draw(qwop.runningTrack())
        phenotype.toDraw = False

    if displayStream:
        cv2.namedWindow(windowName)
        pos = windowPositions[qwop.getNumber()]
        cv2.moveWindow(windowName, pos[0], pos[1])

    differentKeysPressed = []
    startTime = None
    while (running):
        qwop.takeScreenshot()

        if displayStream:
            # qwop.showStream()
            cv2.imshow(windowName, qwop.getImage())
            cv2.waitKey(10)
        
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

    if displayStream:
        cv2.destroyWindow(windowName)
        
    # fitnessScore = max(0, fitnessScore)
    distance = fitnessScore / 100.0
    if (fitnessScore > 0):
        fitnessScore *= len(differentKeysPressed)
    #     fitnessScore = pow(fitnessScore, len(differentKeysPressed))

    return (distance, fitnessScore)

class QueueManager(managers.BaseManager):
    pass # Pass is really enough. Nothing needs to be done here.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("saveFolder")
    parser.add_argument("--load")

    args = parser.parse_args()

    saveDirectory = "saves/" + args.saveFolder
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    sys.setrecursionlimit(10000)

    nrOfOrgamisms = 50

    QWOP(-1).closeAllTabs()

    qwop = QWOP(0)

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

        finishedIndex = 0
        for i, phenotype in enumerate(neat.phenotypes):
            results[i] = testOrganism(phenotype, qwop)
            finishedIndex += 1
            print("\rFinished phenotype ("+ str(finishedIndex) +"/"+ str(len(neat.phenotypes)) +")", end='')

        distances, fitnessScores = results

        for p, d in zip(neat.phenotypes, distances):
            p.genome.distance = d

        print("")
        print("-----------------------------------------------------")
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print(distances)
        print("Generation: " + str(neat.generation))
        print("Number of species: " + str(len(neat.species)))
        table = PrettyTable(["ID", "age", "members", "max fitness", "adj. fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. bias", "avg. compat.", "to spawn"])
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
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        # innovations.printTable()

        # Save current generation
        saveFileName = args.saveFolder + "." + str(neat.generation)
        with open(saveDirectory + "/" + saveFileName, 'wb') as file:
            pickle.dump([neat, innovations], file)

        # Append to summary file
        with open(saveDirectory + "/summary.txt", 'a') as file:
            file.write("Generation " + str(neat.generation) + "\n")
            file.write(table.get_string())
            file.write("\n\n")
