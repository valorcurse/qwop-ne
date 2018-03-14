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

    def __init__(self, speciesID, leader):
        self.ID = speciesID

        self.members = [leader]

        self.age = 0
        self.numToSpawn = 0

        self.leader = leader

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

    def isMember(self, genome):
        # print("members:", [m.ID for m in self.members], "genome:", genome.ID, "->", (genome.ID in [m.ID for m in self.members]))
        return (genome.ID in [m.ID for m in self.members])

    def best(self):
        # return max(self.members, key=attrgetter('fitness'))
        return max(self.members)

    def spawn(self):
        return random.choice(self.members)

    def adjustFitnesses(self, minFitness, maxFitness):
        fitnessRange = max(1.0, maxFitness - minFitness)
        
        # highestFitness = round(highestFitness, 1)

        avgMemberFitness = sum([m.fitness for m in self.members])/len(self.members)
        newAdjustedFitness = (avgMemberFitness - minFitness) / fitnessRange

        if self.age <= self.youngAgeThreshold:
            newAdjustedFitness *= self.youngAgeBonus

        if self.age >= self.oldAgeThreshold:
            newAdjustedFitness *= self.oldAgePenalty

        self.adjustedFitness = newAdjustedFitness

    def becomeOlder(self):
        self.age += 1

        highestFitness = max([m.fitness for m in self.members])

        # Check if species is stagnant
        if (highestFitness < self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement == self.numGensAllowNoImprovement):
            self.stagnant = True

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
        self.maxNumberOfNeuronsPermitted = 20

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
        # self.maxWeightPerturbation = 1.5

        self.activationMutationRate = 0.8
        self.maxActivationPerturbation = 0.8

        inputs = []
        for n in range(numOfInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(numOfInputs) + ")", end='')
            newInput = innovations.createNewNeuron(0.0, NeuronType.INPUT, -n-1)
            inputs.append(newInput)

        print("")

        outputs = []
        for n in range(numOfOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(numOfOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, -numOfInputs-n-1)
            outputs.append(newOutput)

        print("")

        inputs.extend(outputs)
        newGenome = CGenome(self.currentGenomeID, inputs, [], numOfInputs, numOfOutputs)
        for i in range(self.populationSize):
            self.genomes.append(deepcopy(newGenome))
            self.currentGenomeID += 1

        self.speciate()

        self.phenotypes = self.epoch([0]*len(self.genomes))

    def epoch(self, fitnessScores):
        

        if (len(fitnessScores) != len(self.genomes)):
            print("Mismatch of scores/genomes size.")
            return

        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = fitnessScores[index]

        # print("-----------------------------------")
        self.calculateSpawnAmount()
        self.reproduce()
        self.speciate()

        newPhenotypes = []
        for genome in self.genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.generation += 1

        # print([(s.ID, s.numToSpawn, len(s.members)) for s in self.species])
        return newPhenotypes

    def speciate(self):

        # print([(s.ID, s.best().ID, s.best().fitness) for s in self.species])

        # Find best leader for species from the new population
        unspeciated = list(range(0, len(self.genomes)))
        for s in self.species:
            s.members = []
            candidates = []
            for i in unspeciated:
                g = self.genomes[i]
                # print(g)
                distance = g.calculateCompatibilityDistance(s.leader)
                if (distance < self.newSpeciesTolerance):
                    candidates.append((distance, i))

            _, bestCandidate = min(candidates, key=lambda x: x[0])

            s.leader = self.genomes[bestCandidate]
            s.members.append(s.leader)
            unspeciated.remove(bestCandidate)
        
        # Distribute genomes to their closest species
        for i in unspeciated:
            genome = self.genomes[i]

            closestDistance = self.newSpeciesTolerance
            closestSpecies = None
            for s in self.species:
                distance = genome.calculateCompatibilityDistance(s.leader)
                # If genome falls within tolerance of species
                if (distance < closestDistance):
                    closestDistance = distance
                    closestSpecies = s

            if (closestSpecies is not None): # If found a compatible species
                closestSpecies.members.append(genome)

            else: # Else create a new species
                self.speciesNumber += 1
                self.species.append(CSpecies(self.speciesNumber, genome))

        # self.species = [s[0] for s in existingSpecies]
        # print([s.numToSpawn for s in self.species])

    def reproduce(self):
        # print([s.numToSpawn for s in self.species])
        newPop = []
        for s in self.species:
            numToSpawn = s.numToSpawn

            members = deepcopy(s.members)
            members.sort(reverse=True, key=lambda x: x.fitness)

            # print([g.fitness for g in members])
            # print(s.ID, "Best of members:", [(g.ID, g.fitness) for g in members[:2]])
            # Grabbing the top 2 performing genomes
            for topMember in members[:2]:
                newPop.append(topMember)
                members.remove(topMember)
                numToSpawn -= 1



            # Only use the survival threshold fraction to use as parents for the next generation.
            cutoff = int(math.ceil(0.2 * len(members)))
            # Use at least two parents no matter what the threshold fraction result is.
            cutoff = max(cutoff, 2)
            members = members[:cutoff]

            if (numToSpawn <= 0 or len(members) <= 0):
                continue

            # print("Spawning for species:", s.ID, "| Amount:", numToSpawn)
            for i in range(numToSpawn):
                baby = None

                # if (len(members) == 1 or random.random() > self.crossoverRate):
                if (random.random() > self.crossoverRate):
                    # baby = copy(s.spawn())
                    baby = deepcopy(random.choice(members))
                else:
                    # g1 = s.spawn()
                    g1 = random.choice(members)
                    # possibleMates = [g for g in members if g.ID != g1.ID]
                    g2 = random.choice(members)
                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID


                # div = max(1,(self.chanceToAddNode*2 + self.chanceToAddLink*2))
                # r = random.random()
                # if r < (self.chanceToAddNode/div):
                #     baby.addNeuron()
                # elif r < ((self.chanceToAddNode + self.chanceToAddNode)/div):
                #     baby.removeNeuron()
                # elif r < ((self.chanceToAddNode + self.chanceToAddNode +
                #            self.chanceToAddLink)/div):
                #     baby.addLink(self.chanceToAddRecurrentLink,
                #              self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)
                # elif r < ((self.chanceToAddNode + self.chanceToAddNode +
                #            self.chanceToAddLink + self.chanceToAddLink)/div):
                #     baby.removeLink()


                if (random.random() < self.chanceToAddNode):
                    # if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
                    baby.addNeuron()

                # if (random.random() < self.chanceToAddNode):
                    # baby.removeNeuron()

                if (random.random() < self.chanceToAddLink):
                    baby.addLink(self.chanceToAddRecurrentLink, self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

                # if (random.random() < self.chanceToAddLink):
                    # baby.removeLink()


                # elif (random.random() < self.mutationRate):
                baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced, self.maxWeightPerturbation)

                
                baby.mutateBias(self.chanceToMutateBias, self.probabilityOfWeightReplaced, self.maxWeightPerturbation)

                    # baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

                baby.links.sort()
                newPop.append(baby)

        self.genomes = newPop
        # print([s.numToSpawn for s in self.species])

    def calculateSpawnAmount(self):
        # Remove stagnant species
        for s in self.species:
            s.becomeOlder()

            if s.stagnant:
                self.species.remove(s)
        
        # Adjust species fitness
        allFitnesses = [m.fitness for spc in self.species for m in spc.members]
        minFitness = min(allFitnesses) if len(allFitnesses) != 0 else 0.0
        maxFitness = max(allFitnesses) if len(allFitnesses) != 0 else 0.0
        for s in self.species:
            s.adjustFitnesses(minFitness, maxFitness)

        minToSpawn = 2
        sumFitness = sum(s.adjustedFitness for s in self.species)
        spawnAmounts = []
        
        for s in self.species:            
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

        totalSpawn = sum(spawnAmounts)
        norm = self.populationSize / totalSpawn
        spawnAmounts = [max(minToSpawn, int(round(n * norm))) for n in spawnAmounts]
        
        for spawnAmount, species in zip(spawnAmounts, self.species):
            species.numToSpawn = spawnAmount

    def crossover(self, mum, dad):
        
        best = None
        if (mum.fitness == dad.fitness):
            if (len(mum.links) == len(dad.links)):
                best = random.choice([mum, dad])
            else:
                best = mum if len(mum.links) < len(dad.links) else dad
        else:
            best = mum if mum.fitness > dad.fitness else dad

        # Copy input and output neurons
        babyNeurons = [deepcopy(n) for n in best.neurons
                       if (n.neuronType in [NeuronType.INPUT, NeuronType.OUTPUT])]

        combinedIndexes = list(set(
            [l.innovationID for l in mum.links] + [l.innovationID for l in dad.links]))
        combinedIndexes.sort()
        
        mumDict = {l.innovationID: l for l in mum.links}
        dadDict = {l.innovationID: l for l in dad.links}

        # print("-------------------------------------------------")
        babyLinks = []
        for i in combinedIndexes:
            mumLink = deepcopy(mumDict.get(i))
            dadLink = deepcopy(dadDict.get(i))
            
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
                babyNeurons.append(deepcopy(link.fromNeuron))

            if (link.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(deepcopy(link.toNeuron))

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, best.inputs, best.outputs)
