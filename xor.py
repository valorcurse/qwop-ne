from genes import innovations
from neat import NEAT

global innovations

import random
import time
from matplotlib import pyplot
from prettytable import PrettyTable

import numpy as np

import multiprocessing
from multiprocessing import Pool, Queue, Value
import multiprocessing.managers as managers

# import multiprocess as multiprocessing
# from multiprocess.pool import Pool
# from multiprocess import Queue, Value
# import multiprocess.managers as managers

def testOrganism(phenotype, finishedIndex, nrOfPhenotypes):
    if phenotype.toDraw:
        # print("Drawing phenotype", phenotype.ID)
        # phenotype.draw()
        # phenotype.toDraw = False
        print("")
        print("Phenotype", phenotype.ID)

    
    fitnessScore = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = phenotype.update(xi)
        
        fitnessScore -= (output[0] - xo) ** 2
        if phenotype.toDraw:
            print(xi, output[0], "==", xo, "=", fitnessScore)

    if phenotype.toDraw:
        # print("Drawing phenotype", phenotype.ID)
        phenotype.draw()
    phenotype.toDraw = False
    # finishedIndex.value += 1
    # print("\rFinished phenotype ("+ str(finishedIndex.value) +"/"+ str(nrOfPhenotypes) +")", end='')

    return fitnessScore

if __name__ == '__main__':
    print("Creating NEAT object")
    
    highScore = 0.0

    xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    xor_outputs = [0.0, 1.0, 1.0, 0.0]

    neat = NEAT(150, 2, 1)
    # pyplot.ion()
    # pyplot.show(block=False)
    pyplot.show()

    while True:
        largestNetwork = None
        for p in neat.phenotypes:
            if (not largestNetwork):
                largestNetwork = p
            else:
                networkSize = len(p.neurons)
                if (len(p.neurons) > len(largestNetwork.neurons)):
                    largestNetwork = p
        
        # randomPhenotype = random.choice(neat.phenotypes)
        # randomPhenotype.toDraw = True
        # p.toDraw = True

        finishedIndex = multiprocessing.Manager().Value('i', 0)
        # pool = Pool(multiprocessing.cpu_count())
        pool = Pool(4)
        # fitnessScores = pool.map(testOrganism, neat.phenotypes)
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

        # print("")
        # print("-----------------------------------------------------")
        # print(fitnessScores)
        # print("Running epoch")

        print("Number of species: " + str(len(neat.species)))
        table = PrettyTable(["ID", "age", "fitness", "adj. fitness", "distance", "unique keys", "stag", "neurons", "links"])
        for s in neat.species:
            table.add_row([
                s.ID,                                                             # Species ID
                s.age,                                                            # Age
                int(max([m.fitness for m in s.members])),                         # Average fitness
                "{:1.4f}".format(s.adjustedFitness),                              # Adjusted fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),           # Average distance
                "{:1.4f}".format(max([m.uniqueKeysPressed for m in s.members])),  # Average unique keys
                s.generationsWithoutImprovement,                                  # Stagnation
                int(np.mean([len(m.neurons) for m in s.members])),           # Neurons
                np.mean([len(m.links) for m in s.members])])                      # Links

        print(table)

        neat.phenotypes = neat.epoch(fitnessScores)
        print("")
        print("####################### Generation: " + str(neat.generation) + " #######################")
        # print("Number of innovations: " + str(len(innovations.listOfInnovations)))
        allFitnesses = sum([g.fitness for g in neat.genomes])
        # print("Best fitness:", max(fitnessScores))
        # print("Average fitness:", allFitnesses/len(neat.genomes))
        # print("Number of phenotypes: " + str(len(neat.phenotypes)))
        # print("Number of species: " + str(len(neat.species)))
        # print("ID", "\t", "age", "\t", "fitness", "\t", "adj. fitness", "\t", "stag")
        # for s in neat.species:
        #     print(s.ID, "\t", s.age, "\t", "{:1.4f}".format(max([m.fitness for m in s.members])), 
        #         "\t", "{:1.4f}".format(s.adjustedFitness), "\t", s.generationsWithoutImprovement)


        time.sleep(0.1)
        