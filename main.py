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

import numpy as np
import cv2
import time

import os
import cProfile

def profiler(phenotype):
    cProfile.runctx('testOrganism(phenotype)', globals(), locals(), 'prof.prof')

def testOrganism(phenotype):
    running = True
    gameStarted = False

    fitnessScore = 0

    startTime = None

    qwop = None
    introStartTime = time.time()
    while (qwop == None or not qwop.isAtIntro()):
        try:
            qwop = QWOP()
        except TimeoutException:
            qwop.stop()
            continue

        qwop.grabImage()
        if (time.time() - introStartTime) > 30.0:
            print("Restarting QWOP instance.")
            introStartTime = time.time()
            qwop.stop()

    # print("Network:")
    # print("Hidden neurons: " +
    #   str(len([neuron for neuron in phenotype.neurons if neuron.neuronType == NeuronType.HIDDEN])))

    while (running):
        qwop.grabImage()

        if (not qwop.isPlayable()):
            if (not gameStarted):
                gameStarted = True
                qwop.startGame()
            else:
                running = False
        else:
            previousFitnessScore = fitnessScore
            fitnessScore = qwop.score()

            if fitnessScore == previousFitnessScore:
                if startTime == None:
                    startTime = time.time()
                else:
                    # print("\rTime standing still: " + str(time.time() - startTime), end='')
                    if (time.time() - startTime) > 3.0:
                        # print("")
                        # print("Stopping game.")
                        running = False
            else:
                startTime = None

            predicted = np.argmax(phenotype.update(qwop.runningTrack().flatten()), axis=0)

            qwop.pressKey(Key(predicted).name)

    # print("")
    # print("Fitness score: " + str(fitnessScore))
    # fitnessScores.append(fitnessScore)
    qwop.stop()

    return fitnessScore


if __name__ == '__main__':
    qwop = QWOP()
    # net = Net()
    # net.cuda()

    qwop.grabImage()
    # cv2.imshow('running track', qwop.runningTrack())
    # cv2.waitKey()

    print("Creating NEAT object")
    nrOfOrgamisms = 20
    neat = NEAT(nrOfOrgamisms, qwop.runningTrack().size, 4)
    qwop.stop()

    multiprocessing.set_start_method('spawn')
    while True:
        # fitnessScores = []

        # pool = Pool(nrOfOrgamisms)
        pool = Pool(5)
        fitnessScores = pool.map(testOrganism, neat.phenotypes)
        # fitnessScores = pool.map(profiler, neat.phenotypes)
        pool.close()
        pool.join()


        print("-----------------------------------------------------")
        print(fitnessScores)
        print("Running epoch")
        neat.phenotypes = neat.epoch(fitnessScores)
        print("Generation: " + str(neat.generation))
        print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        print("Number of genomes: " + str(len(neat.genomes)))
        print("Number of species: " + str(len(neat.species)))
        print("Number of phenotypes: " + str(len(neat.phenotypes)))
