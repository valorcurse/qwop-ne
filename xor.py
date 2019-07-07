from neat.neat import NEAT
from neat.genes import MutationRates

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

import cProfile
import operator

xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
# xor_outputs = [0.0, 1.0, 1.0, 0.0]
xor_outputs = [0.0, 1.0, 1.0, 0.0]


def testOrganism(phenotype, finishedIndex, nrOfPhenotypes, outputs=None):
    
    # print("")
    # print("Phenotype", phenotype.ID)
    
    fitnessScore = 4.0
    answers = []
    for xi, xo in zip(xor_inputs, xor_outputs):
        # output = phenotype.update(xi)
        output = phenotype.updateRecursively(xi)

        fitnessScore -= (output[0] - xo) ** 2

        # if phenotype.toDraw:
            # print("Drawing phenotype", phenotype.ID)
            # phenotype.draw()
            
        # fitnessScore -= math.fabs(xo - output[0])
        answers.append(output)

    # phenotype.toDraw = False
    fitnessScore = round(fitnessScore, 5)
    # outputs.append((fitnessScore, answers))
    return (fitnessScore, answers)

def profileOrganism(phenotype, finishedIndex, nrOfPhenotypes):
    outputs = []
    cProfile.runctx('testOrganism(phenotype, finishedIndex, nrOfPhenotypes, outputs)', globals(), locals(), 'prof.prof')
    return outputs[0]

if __name__ == '__main__':
    print("Creating NEAT object")

    highScore = 0.0
    highScoreAnswers = []

    neat = NEAT(300, 2, 1)
    mutationRates = MutationRates()
    mutationRates.newSpeciesTolerance = 3.0
    mutationRates.maxWeightPerturbation = 0.1
    mutationRates.chanceToAddNeuron = 0.1
    mutationRates.chanceToAddLink = 0.4
    
    mutationRates.chanceToMutateBias = 0.4
    mutationRates.mutationRate = 0.75
    mutationRates.crossoverRate = 0.75
    
    # mutationRates.mpcStagnation = 10
    mutationRates.mpcMargin = 50

    neat = NEAT(300, 2, 1, mutationRates, False)

    while True:
        # randomPhenotype = random.choice(neat.phenotypes)
        # randomPhenotype.toDraw = True
        # print([(s.ID, s.numToSpawn, len(s.members)) for s in neat.species])

        finishedIndex = multiprocessing.Manager().Value('i', 0)
        pool = Pool(4)
        results = {}
        answers = {}
        for i, phenotype in enumerate(neat.phenotypes):
            # results[i] = pool.apply_async(testOrganism, (phenotype, finishedIndex, len(neat.phenotypes)))
            # results[i] = pool.apply_async(profileOrganism, (phenotype, finishedIndex, len(neat.phenotypes)))
            results[i], answers[i] = testOrganism(phenotype, finishedIndex, len(neat.phenotypes))

        pool.close()
        pool.join()

        # print(results.items())
        fitnessScores = results
        # fitnessScores = [result.get()[0] for func, result in results.items()]
        # answers = [result[1] for result in results]
        # answers = [result.get()[1] for func, result in results.items()]
        # answers = [a[0] for a in answers]
        # fitnessScores = [result[0] for i, result in results.items()]
        # answers = [result[1] for i, result in results.items()]
        # [item for (item,) in x]
        # print(fitnessScores)
        # print(answers)
    
        # print("\n")

        for index, genome in enumerate(neat.population.genomes):
            genome.fitness = results[index]

        # highestFitness = max(fitnessScores)
        print(fitnessScores)
        index, highestFitness = max(fitnessScores.items())
        if (highestFitness > highScore):
            highScore = highestFitness
            highScoreAnswers = answers[index]

            if (highScore > 3.9):
                print("Reached 3.9!")
                # wonTable = PrettyTable(["ID", "age", "members", "max fitness", "adj. fitness", "stag", "neurons", "links", "avg. weight", "avg. bias", "to spawn"])
                # wonTable.add_row([
                #     s.ID,                                                       # Species ID
                #     s.age,                                                      # Age
                #     len(s.members),                                             # Nr. of members
                #     max([m.fitness for m in s.members]),                        # Max fitness
                #     "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                #     s.generationsWithoutImprovement,                            # Stagnation
                #     "{:1.4f}".format(np.mean([len(m.neurons) for m in s.members])), # Neurons
                #     "{:1.4f}".format(np.mean([len(m.links) for m in s.members])),                 # Links
                #     "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                #     "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                #     s.numToSpawn])                                              # Nr. of members to spawn
                # print(wonTable)
                # neat.phenotypes[index].draw()
                break

            print("###########################")
            print("NEW HIGHSCORE", highScore)
            print("###########################")

        print("")
        print("####################### Generation: " + str(neat.generation) + " #######################")

        print("Phase: " + str(neat.phase))
        # print("MPC: (%s/%s)" % (neat.calculateMPC(), neat.mpcThreshold))
        print("Number of genomes: " + str(len(neat.population.genomes)))
        print("Number of species: " + str(len(neat.population.species)))
        print("Highest score:", highScore)
        answersTable = PrettyTable(["[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"])
        answersTable.add_row(xor_outputs)
        print(highScoreAnswers)
        answersTable.add_row(highScoreAnswers)
        print(answersTable)
        # print("Correct answers:", xor_outputs)
        # print("Highest score answers:", highScoreAnswers)
        table = PrettyTable(["ID", "age", "members", "max fitness", "stag", "neurons", "links", "avg. weight", "avg. bias", "to spawn"])
        for s in neat.population.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                "{:1.4f}".format(max([m.fitness for m in s.members])),                        # Max fitness
                # "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                s.generationsWithoutImprovement,                            # Stagnation
                "{:1.4f}".format(np.mean([len(m.neurons) for m in s.members])), # Neurons
                "{:1.4f}".format(np.mean([len(m.links) for m in s.members])),                 # Links
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                s.numToSpawn])                                              # Nr. of members to spawn
            # print(s.ID, [m.ID for m in s.members])

        print(table)
        # print([(s.ID, s.numToSpawn, len(s.members)) for s in neat.species])

        neat.epoch([r for r in results])

        # time.sleep(1.5)
        