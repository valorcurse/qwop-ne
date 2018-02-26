from genes import innovations
from neat import NEAT

global innovations

import random
import time
from matplotlib import pyplot
from prettytable import PrettyTable

import numpy as np
import math

import multiprocessing
from multiprocessing import Pool, Queue, Value
import multiprocessing.managers as managers

xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_outputs = [0.0, 1.0, 1.0, 0.0]

def testOrganism(phenotype, finishedIndex, nrOfPhenotypes):
    # if phenotype.toDraw:
        # print("Drawing phenotype", phenotype.ID)
        # phenotype.draw()
        # phenotype.toDraw = False
    # print("")
    # print("Phenotype", phenotype.ID)
    
    fitnessScore = 4.0
    answers = []
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = phenotype.update(xi)
        
        # fitnessScore -= abs(output[0] - xo) ** 2
        score = math.fabs(xo - output[0])
        fitnessScore -= score
        answers.append(score)
        # if phenotype.toDraw:
        # print(xi, output[0], "==", xo, "=", fitnessScore)

    # if phenotype.toDraw:
    nrOfLinks = [n for n in phenotype.neurons for l in n.linksIn]
    # print(len(nrOfLinks))
    if fitnessScore == 2.0 and len(phenotype.neurons) == 5 and len(nrOfLinks) == 7:

        # print("Drawing phenotype", phenotype.ID)
        # print("Number of species: " + str(len(neat.species)))
        print("Displaying Phenotype:", phenotype.ID)
        print(answers)
        print(fitnessScore)
        for n in phenotype.neurons:
            print("Neuron", n.ID)
            table = PrettyTable(["from", "to", "weight"])
            for l in n.linksIn:
                table.add_row([
                    l.fromNeuron.ID,
                    l.toNeuron.ID,
                    l.weight])
            print(table)

        phenotype.draw()

        for xi, xo in zip(xor_inputs, xor_outputs):
            print(xi, "->", phenotype.update(xi))


    phenotype.toDraw = False
    # finishedIndex.value += 1
    # print("\rFinished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")", end='')

    return fitnessScore

if __name__ == '__main__':
    print("Creating NEAT object")
    
    highScore = 0.0

    neat = NEAT(150, 2, 1)
    # pyplot.show()

    while True:
        randomPhenotype = random.choice(neat.phenotypes)
        randomPhenotype.toDraw = True

        finishedIndex = multiprocessing.Manager().Value('i', 0)
        pool = Pool(4)
        results = {}
        for i, phenotype in enumerate(neat.phenotypes):
            results[i] = pool.apply_async(testOrganism, (phenotype, finishedIndex, len(neat.phenotypes)))
        pool.close()
        pool.join()

        fitnessScores = [result.get() for func, result in results.items()]
    
        print("\n")

        highestFitness = max(fitnessScores)
        if (highestFitness > highScore):
            highScore = highestFitness

            print("###########################")
            print("NEW HIGHSCORE", highScore)
            print("###########################")

        neat.phenotypes = neat.epoch(fitnessScores)
        print("")
        print("####################### Generation: " + str(neat.generation) + " #######################")

        print("Number of species: " + str(len(neat.species)))
        table = PrettyTable(["ID", "age", "members", "max fitness", "adj. fitness", "stag", "neurons", "links", "to spawn"])
        for s in neat.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                max([m.fitness for m in s.members]),                        # Max fitness
                "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                s.generationsWithoutImprovement,                            # Stagnation
                int(np.mean([len(m.neurons) for m in s.members])),          # Neurons
                np.mean([len(m.links) for m in s.members]),                 # Links
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        time.sleep(0.1)
        