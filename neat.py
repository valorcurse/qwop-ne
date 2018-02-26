import random
from random import randint

import math
import numpy as np
from enum import Enum

from copy import copy
from copy import deepcopy
import itertools
from operator import attrgetter

from prettytable import PrettyTable

import genes
from genes import NeuronType
from genes import CGenome
from genes import innovations

global innovations

class CSpecies:
    numGensAllowNoImprovement = 20

    def __init__(self, speciesID):
        self.ID = speciesID

        self.members = []
        self.age = 0
        self.numToSpawn = 0

        self.youngAgeThreshold = 10
        self.youngAgeBonus = 1.5
        self.oldAgeThreshold = 50
        self.oldAgePenalty = 0.5

        self.highestFitness = 0.0
        self.adjustedFitness = 0.0
        self.generationsWithoutImprovement = 0

        self.stagnant = False

    def __contains__(self, key):
        return key.ID in [m.ID for m in self.members]

    def leader(self):
        # return max(self.members, key=attrgetter('fitness'))
        return max(self.members)

    def spawn(self):
        return random.choice(self.members)

    def adjustFitnesses(self, minFitness, maxFitness):
        fitnessRange = max(1.0, maxFitness - minFitness)
        
        highestFitness = max([m.fitness for m in self.members])
        # highestFitness = round(highestFitness, 1)

        avgMemberFitness = sum([m.fitness for m in self.members])/len(self.members)
        newAdjustedFitness = (avgMemberFitness - minFitness) / fitnessRange

        # Check if species is stagnant
        if (highestFitness < self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement == self.numGensAllowNoImprovement):
            self.stagnant = True

        if self.age <= self.youngAgeThreshold:
            newAdjustedFitness *= self.youngAgeBonus

        if self.age >= self.oldAgeThreshold:
            newAdjustedFitness *= self.oldAgePenalty

        self.adjustedFitness = newAdjustedFitness

class NEAT:
    def __init__(self, numberOfGenomes, numOfInputs, numOfOutputs):

        self.genomes = []
        self.phenotypes = []
        self.species = []
        self.speciesNumber = 0

        self.populationSize = numberOfGenomes

        self.generation = 0

        self.currentGenomeID = 0

        self.crossoverRate = 0.7
        self.maxNumberOfNeuronsPermitted = 5

        # self.newSpeciesTolerance = 5.0
        self.newSpeciesTolerance = 3.0

        self.chanceToMutateBias = 0.7

        self.chanceToAddNode = 0.2
        # self.chanceToAddNode = 1.0
        self.numOfTriesToFindOldLink = 10

        self.chanceToAddLink = 0.5
        self.chanceToAddRecurrentLink = 0.05
        self.numOfTriesToFindLoopedLink = 15
        self.numOfTriesToAddLink = 20

        self.mutationRate = 0.8
        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.activationMutationRate = 0.8
        self.maxActivationPerturbation = 0.8

        inputs = []
        # print("Creating input neurons:")
        for n in range(numOfInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(numOfInputs) + ")", end='')
            newInput = innovations.createNewNeuron(None, None, n, 0.0, NeuronType.INPUT)
            # print("neuron id:", newInput.ID)
            inputs.append(newInput)

        print("")

        biasInput = innovations.createNewNeuron(None, None, n, 0.0, NeuronType.BIAS)
        inputs.append(biasInput)

        outputs = []
        for n in range(numOfOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(numOfOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(None, None, n, 1.0, NeuronType.OUTPUT)
            outputs.append(newOutput)

        print("")
        inputs.extend(outputs)

        for i in range(self.populationSize):
            newGenome = CGenome(self.currentGenomeID, inputs, [], numOfInputs, numOfOutputs)
            self.genomes.append(newGenome)
            # newSpecies.members.append(newGenome)

            self.currentGenomeID += 1

        self.phenotypes = self.epoch([0]*len(self.genomes))

    def epoch(self, fitnessScores):
        if (len(fitnessScores) != len(self.genomes)):
            print("Mismatch of scores/genomes size.")
            return

        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = fitnessScores[index]

        # Empty species except for best performing genome
        for s in self.species:
            leader = s.leader()
            s.members = [leader]

        
        
        # Distribute genomes to their closest species
        for genome in self.genomes:
            speciesMatched = False

            table = PrettyTable(["leader ID", "species ID", "distance", "matched"])
            # matchedTable = PrettyTable(["ID", "neurons", "links", "avg. weigths", "weigths"])
            matchedTable = PrettyTable(["ID", "neurons", "links", "avg. weigths", "distance"])

            # for s in self.species:
                # if genome not in :
                    # print(s.leader().ID, genome.ID)

            # otherSpecies = [s for s in self.species if genome not in s.members]
            # print([s.leader().ID for s in otherSpecies])
            for s in self.species:
                leader = s.leader()

                if (genome == leader):
                    speciesMatched = True
                    break

                distance = genome.calculateCompatibilityDistance(leader)

                # print(genome.ID, "->", leader.ID, "=", distance)
                # print("{} ({},{}) -> {} ({}, {}) = {}".format(genome.ID, len(genome.neurons), len(genome.links), leader.ID, len(leader.neurons), len(leader.links), distance))

                # If genome falls within tolerance of species, add it
                if (distance < self.newSpeciesTolerance):
                    s.members.append(genome)
                    speciesMatched = True
                    table.add_row([leader.ID, s.ID, distance, speciesMatched])

                    # matchedTable.add_row([genome.ID, len(genome.neurons), len(genome.links), np.mean([l.weight for l in genome.links]), [l.weight for l in genome.links]])
                    # matchedTable.add_row([leader.ID, len(leader.neurons), len(leader.links), np.mean([l.weight for l in leader.links]), [l.weight for l in leader.links]])

                    matchedTable.add_row([genome.ID, len(genome.neurons), len(genome.links), np.mean([l.weight for l in genome.links]), distance])
                    matchedTable.add_row([leader.ID, len(leader.neurons), len(leader.links), np.mean([l.weight for l in leader.links]), distance])

                    break

                table.add_row([leader.ID, s.ID, distance, False])

            # print("Genome", genome.ID)
            # print(table)

            # Else create a new species
            if (not speciesMatched):
                self.speciesNumber += 1

                newSpecies = CSpecies(self.speciesNumber)
                newSpecies.members.append(genome)
                self.species.append(newSpecies)
            # else:
                # print("Matching genomes:")
                # print(matchedTable)
            

        # Adjust species fitness
        allFitnesses = [m.fitness for spc in self.species for m in spc.members]
        minFitness = min(allFitnesses)
        maxFitness = max(allFitnesses)
        for s in self.species:
            s.adjustFitnesses(minFitness, maxFitness)
            s.age += 1

        # Calculate number of spawns
        minToSpawn = 2
        sumFitness = sum(s.adjustedFitness for s in self.species)
        spawnAmounts = []
        for s in self.species:
            if s.stagnant:
                self.species.remove(s)
                continue
            
            if (sumFitness > 0):
                size = max(minToSpawn, s.adjustedFitness / sumFitness * self.populationSize)
            else:
                size = minToSpawn

            previousSize = len(s.members)
            sizeDifference = (size - previousSize) * 0.5
            roundedSize = int(round(sizeDifference))
            toSpawn = previousSize
            if abs(roundedSize) > 0:
                toSpawn += roundedSize
            elif sizeDifference > 0:
                toSpawn += 1
            elif sizeDifference < 0:
                toSpawn -= 1

            spawnAmounts.append(toSpawn)

        totalSpawn = max(minToSpawn, sum(spawnAmounts))
        norm = self.populationSize / totalSpawn
        spawnAmounts = [max(minToSpawn, int(round(n * norm))) for n in spawnAmounts]
        for spawnAmount, species in zip(spawnAmounts, self.species):
            species.numToSpawn = spawnAmount

        newPop = []
        for s in self.species:
            chosenBestYet = False

            numToSpawn = s.numToSpawn
            # print("Spawning for species:", s.ID, "| Amount:", numToSpawn)
            for i in range(numToSpawn):
                baby = None

                if (not chosenBestYet):
                    baby = s.leader()
                    chosenBestYet = True

                else:
                    if (len(s.members) == 1 or random.random() > self.crossoverRate):
                        baby = deepcopy(s.spawn())
                    else:
                        g1 = s.spawn()
                        possibleMates = [g for g in s.members if g.ID != g1.ID]
                        g2 = random.choice(possibleMates)
                        baby = self.crossover(g1, g2)

                    self.currentGenomeID += 1
                    baby.ID = self.currentGenomeID

                    if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
                        baby.addNeuron(self.chanceToAddNode)

                    # for i in range(15):
                    baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
                                 self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

                    baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
                                       self.maxWeightPerturbation)

                    baby.mutateBias(self.chanceToMutateBias, self.maxWeightPerturbation)

                    # baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

                baby.links.sort()
                newPop.append(baby)

        self.genomes = newPop

        newPhenotypes = []
        for genome in self.genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.generation += 1

        return newPhenotypes

    def crossover(self, mum, dad):
        
        best = None
        if (mum.fitness == dad.fitness):
            if (len(mum.links) == len(dad.links)):
                best = random.choice([mum, dad])
            else:
                best = mum if len(mum.links) < len(dad.links) else dad
        else:
            best = mum if mum.fitness > dad.fitness else dad

        # Copy input, bias and output neurons
        babyNeurons = [n for n in best.neurons
                       if (n.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.OUTPUT])]

        combinedIndexes = list(set(
            [l.innovationID for l in mum.links] + [l.innovationID for l in dad.links]))
        combinedIndexes.sort()
        
        mumDict = {l.innovationID: l for l in mum.links}
        dadDict = {l.innovationID: l for l in dad.links}

        # print("-------------------------------------------------")
        babyLinks = []
        for i in combinedIndexes:
            mumLink = mumDict.get(i)
            dadLink = dadDict.get(i)
            
            # print(mumLink.innovationID if mumLink else "None", dadLink.innovationID if dadLink else "None")
            
            if (mumLink is None):
                if (best == dad):
                    # print("mum is None and best is dad")
                    babyLinks.append(dadLink)

            elif (dadLink is None):
                if (best == mum):
                    # print("dad is None and best is mum")
                    babyLinks.append(mumLink)

            else:
                babyLinks.append(random.choice([mumLink, dadLink]))

        # print("")
        for link in babyLinks:
            # print(link.innovationID)

            if (link.fromNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(link.fromNeuron)

            if (link.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(link.toNeuron)

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, best.inputs, best.outputs)
