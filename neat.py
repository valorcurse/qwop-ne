import random
from random import randint

import math
from enum import Enum

from copy import copy
import itertools
from operator import attrgetter

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

        self.youngBonusAgeThreshold = 10
        self.youngFitnessBonus = 1.5
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
        avgMemberFitness = sum([m.fitness for m in self.members])/len(self.members)
        newAdjustedFitness = (avgMemberFitness - minFitness) / fitnessRange

        # Check if species is stagnant
        if (highestFitness <= self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement == self.numGensAllowNoImprovement):
            self.stagnant = True

        self.adjustedFitness = newAdjustedFitness

class NEAT:
    genomes = []
    phenotypes = []
    species = []
    speciesNumber = 0

    populationSize = 25

    generation = 0

    currentGenomeID = 0

    numOfSweepers = None
    crossoverRate = 0.7
    maxNumberOfNeuronsPermitted = 15

    newSpeciesTolerance = 3.0

    chanceToMutateBias = 0.7

    chanceToAddNode = 0.03
    numOfTriesToFindOldLink = 10

    chanceToAddLink = 0.07
    chanceToAddRecurrentLink = 0.05
    numOfTriesToFindLoopedLink = 15
    numOfTriesToAddLink = 20

    mutationRate = 0.2
    probabilityOfWeightReplaced = 0.1
    maxWeightPerturbation = 0.5

    activationMutationRate = 0.8
    maxActivationPerturbation = 0.8

    def __init__(self, numberOfGenomes, numOfInputs, numOfOutputs):
        self.numOfSweepers = numberOfGenomes

        # newSpecies = CSpecies(self.speciesNumber)

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

        for i in range(self.numOfSweepers):
            newGenome = CGenome(self.currentGenomeID, inputs, [], numOfInputs, numOfOutputs)
            self.genomes.append(newGenome)
            # newSpecies.members.append(newGenome)

            self.currentGenomeID += 1

        # self.phenotypes = self.epoch([0] * len(self.genomes))
        for genome in self.genomes:
            # depth = calculateNetDepth(genome

            depth = len(set(n.splitY for n in genome.neurons))
            phenotype = genome.createPhenotype(depth)

            self.phenotypes.append(phenotype)

    def epoch(self, fitnessScores):
        if (len(fitnessScores) != len(self.genomes)):
            print("Mismatch of scores/genomes size.")
            return

        # print("Number of species:", len(self.species))
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

            speciesIndex = 0
            while (speciesIndex < len(self.species)):
                s = self.species[speciesIndex]
                leaders = [spc.leader() for spc in self.species]

                if (genome in leaders):
                    speciesMatched = True
                    break

                distance = genome.calculateCompatibilityDistance(s.leader())
                # If genome falls within tolerance of species, add it
                if (distance < self.newSpeciesTolerance):
                    s.members.append(genome)
                    speciesMatched = True
                    break

                speciesIndex += 1

            # Else create a new species
            if (not speciesMatched):
                self.speciesNumber += 1

                newSpecies = CSpecies(self.speciesNumber)
                newSpecies.members.append(genome)
                self.species.append(newSpecies)

        # Adjust species fitness
        allFitnesses = [m.fitness for spc in self.species for m in spc.members]
        minFitness = min(allFitnesses)
        maxFitness = max(allFitnesses)
        for s in self.species:
            leader = s.leader()
            s.age += 1
            s.adjustFitnesses(minFitness, maxFitness)

        # Calculate number of spawns
        minToSpawn = 2
        sumFitness = sum(s.adjustedFitness for s in self.species)
        spawnAmounts = []
        for s in self.species:
            if s.stagnant:
                self.species.remove(s)
                continue

            toSpawn = 0
            
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
            print("Spawning for species:", s.ID, "| Amount:", numToSpawn)
            for i in range(numToSpawn):
                baby = None

                if (not chosenBestYet):
                    baby = s.leader()
                    chosenBestYet = True

                else:
                    if (len(s.members) == 1 or random.random() > self.crossoverRate):
                        baby = copy(s.spawn())
                    else:
                        g1 = s.spawn()
                        possibleMates = [g for g in s.members if g.ID != g1.ID]
                        g2 = random.choice(possibleMates)
                        baby = self.crossover(g1, g2)

                    self.currentGenomeID += 1
                    baby.ID = self.currentGenomeID

                    if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
                        baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

                    baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
                                 self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

                    baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
                                       self.maxWeightPerturbation)

                    baby.mutateBias(self.chanceToMutateBias, self.maxWeightPerturbation)

                    # baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

                baby.links.sort()
                newPop.append(baby)
                numToSpawn -= 1

        self.genomes = newPop

        newPhenotypes = []

        for genome in self.genomes:
            depth = len(set(n.splitY for n in genome.neurons))
            phenotype = genome.createPhenotype(depth)

            newPhenotypes.append(phenotype)

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

        babyNeurons = [n for n in mum.neurons
                       if (n.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.OUTPUT])]

        babyLinks = []
        selectedLink = None
        mumIt = dadIt = 0

        combinedLinks = itertools.zip_longest(mum.links, dad.links)
        for mumLink, dadLink in combinedLinks:
            if (not mumLink):
                if (best == dad):
                    selectedLink = dadLink

            elif (not dadLink):
                if (best == mum):
                    selectedLink = mumLink

            elif (dadLink.innovationID == mumLink.innovationID):
                selectedLink = random.choice([mumLink, dadLink])

            elif (mumLink.innovationID < dadLink.innovationID):
                if (best == mum):
                    selectedLink = mumLink

            elif (dadLink.innovationID < mumLink.innovationID):
                if (best == dad):
                    selectedLink = dadLink


            if (selectedLink != None):
                if (len(babyLinks) == 0):
                    babyLinks.append(selectedLink)
                else:
                    if (selectedLink.innovationID not in [l.innovationID for l in babyLinks]):
                        babyLinks.append(selectedLink)

                if (selectedLink.fromNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                    babyNeurons.append(selectedLink.fromNeuron)

                if (selectedLink.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                    babyNeurons.append(selectedLink.toNeuron)

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, mum.inputs, mum.outputs)
