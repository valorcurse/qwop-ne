import random
from random import randint

import math
from enum import Enum

from copy import copy

from operator import attrgetter

import genes
from genes import NeuronType
from genes import CGenome
from genes import innovations

global innovations

class CSpecies:
    # TODO: Find out correct values
    youngBonusAgeThreshold = 5
    youngFitnessBonus = 1.5
    # youngFitnessBonus = 1.0
    oldAgeThreshold = 15
    oldAgePenalty = 0.5
    # oldAgePenalty = 1.0

    avgFitness = 0.0
    numOfGensWithoutImprovement = 0

    def __init__(self, speciesID):
        self.ID = speciesID

        self.members = []
        self.age = 0
        self.numToSpawn = 0

    def __contains__(self, key):
        return key.ID in [m.ID for m in self.members]

    def leader(self):
        # return max(self.members, key=attrgetter('fitness'))
        return max(self.members)

    def spawn(self):
        return random.choice(self.members)

    def adjustFitnesses(self):
        total = 0.0

        for member in self.members:
            fitness = member.fitness

            if (self.age < self.youngBonusAgeThreshold):
                fitness *= self.youngFitnessBonus

            if (self.age > self.oldAgeThreshold):
                fitness *= self.oldAgePenalty

            adjustedFitness = fitness / len(self.members)

            total += adjustedFitness

            member.adjustedFitness = adjustedFitness

        self.avgFitness = total / len(self.members)


class NEAT:
    genomes = []
    phenotypes = []
    species = []
    speciesNumber = 0

    generation = 0
    numGensAllowNoImprovement = 20

    currentGenomeID = 0

    numOfSweepers = None
    crossoverRate = 0.7
    # crossoverRate = 0.5
    # maxNumberOfNeuronsPermitted = 15
    maxNumberOfNeuronsPermitted = 10000

    # newSpeciesTolerance = 3.0
    newSpeciesTolerance = 5

    chanceToAddNode = 0.1
    # chanceToAddNode = 0.03
    numOfTriesToFindOldLink = 10

    # chanceToAddLink = 0.05
    chanceToAddLink = 0.3
    chanceToAddRecurrentLink = 0.01
    numOfTriesToFindLoopedLink = 15
    numOfTriesToAddLink = 20

    mutationRate = 0.8
    probabilityOfWeightReplaced = 0.01
    maxWeightPerturbation = 0.9

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

        # biasInput = innovations.createNewNeuron(None, None, n, 0.0, NeuronType.BIAS)
        # inputs.append(biasInput)

        outputs = []
        for n in range(numOfOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(numOfOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(None, None, n, 1.0, NeuronType.OUTPUT)
            outputs.append(newOutput)

        print("")
        links = []
        # i = 0
        # for outputNeuron in outputs:
        #     # inputNeuron = inputs[0]
        #     for inputNeuron in inputs:
        #         print("\rCreating links (" + str(i + 1) + "/" + str(len(outputs) * len(inputs)) + ")", end='')
        #         # newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, 0.0)
        #         if (inputNeuron.neuronType != NeuronType.BIAS):
        #             newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, 0.0)
        #             links.append(newLink)
        
        #         i += 1

        print("")
        inputs.extend(outputs)
        # print("created neurons:")
        # for neuron in inputs:
        # print("neuron ID:", neuron.ID)

        for i in range(self.numOfSweepers):
            newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
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

        print("Number of species:", len(self.species))
        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = fitnessScores[index]

        # Empty species except for best performing genome
        for s in self.species:
            leader = s.leader()
            s.members = [leader]

        print("Total number of genomes: ", len(self.genomes))
        print("Total number of species: ", len(self.species))
        # Distribute genomes to their closest species
        for genome in self.genomes:
            speciesMatched = False
            print("Genome:", genome.ID, 
                "| Hidden neurons:", len([n for n in genome.neurons if n.neuronType == NeuronType.HIDDEN]),
                "| Links:", len(genome.links))

            speciesIndex = 0
            while (speciesIndex < len(self.species)):
                s = self.species[speciesIndex]
                leaders = [spc.leader() for spc in self.species]
                # print("Leaders: ", [[l.ID, l] for l in leaders])

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

        # print("Number of species: " + str(len(self.species)))
        for s in self.species:
            leader = s.leader()
            s.age += 1
            s.adjustFitnesses()

        minToSpawn = 25
        for s in self.species:
            avgFitness = max(1.0, sum([m.adjustedFitness for m in s.members]) / len(s.members))
            sumAdjustedFitness = sum([m.adjustedFitness for m in s.members])

            if (avgFitness <= s.avgFitness):
                s.numOfGensWithoutImprovement += 1

                if (s.numOfGensWithoutImprovement == self.numGensAllowNoImprovement):
                    self.species.remove(s)
                    continue
            else:
                s.numOfGensWithoutImprovement = 0

            toSpawn = 0
            for member in s.members:
                # toSpawn += member.adjustedFitness / avgFitness
                toSpawn = 0
                if sumAdjustedFitness > 0:
                    toSpawn = max(minToSpawn, member.adjustedFitness / sumAdjustedFitness)
                else:
                    toSpawn = minToSpawn

                # toSpawn += member.adjustedFitness

            s.numToSpawn = max(minToSpawn, toSpawn)

        newPop = []

        for s in self.species:
            numSpawnedSoFar = 0
            if (numSpawnedSoFar < self.numOfSweepers):
                numToSpawn = round(s.numToSpawn)
                print("Spawning for species:", s.ID, "| Amount:", numToSpawn)
                chosenBestYet = False

                for i in range(numToSpawn):
                    baby = None

                    if (not chosenBestYet):
                        baby = s.leader()
                        chosenBestYet = True

                    else:
                        if (len(s.members) == 1):
                            baby = copy(s.spawn())
                        else:
                            g1 = s.spawn()

                            if (random.random() < self.crossoverRate):
                                g2 = s.spawn()

                                numOfAttempts = 5

                                while ((g1.ID == g2.ID) and (numOfAttempts > 0)):
                                    numOfAttempts -= 1
                                    g2 = s.spawn()

                                if (g1.ID != g2.ID):
                                    baby = self.crossover(g1, g2)
                                else:
                                    baby = copy(g1)
                            else:
                                baby = copy(g1)

                        self.currentGenomeID += 1
                        baby.ID = self.currentGenomeID

                        # for i in range(5):
                        if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
                            baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

                        # for i in range(5):
                        baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
                                     self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

                        baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
                                           self.maxWeightPerturbation)

                    # baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

                    baby.links.sort()

                    newPop.append(baby)

                    # print("New pop:", [g.ID for g in newPop])
                    numSpawnedSoFar += 1

        # print("newpop:", len(newPop))
        # if (numSpawnedSoFar < self.numOfSweepers):
            # requiredNumberOfSpawns = self.numOfSweepers - numSpawnedSoFar

        # for i in requiredNumberOfSpawns:
        # newPop.append(TournamentSelection())

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
        while (mumIt < len(mum.links) or dadIt < len(dad.links)):

            if mumIt < len(mum.links):
                currentMum = mum.links[mumIt]

            if dadIt < len(dad.links):
                currentDad = dad.links[dadIt]

            if (mumIt == (len(mum.links)) and (dadIt != len(dad.links))):
                if (best == dad):
                    selectedLink = currentDad

                dadIt += 1

            elif (dadIt == (len(dad.links)) and (mumIt != len(mum.links))):
                if (best == mum):
                    selectedLink = currentMum

                mumIt += 1

            elif (currentDad.innovationID == currentMum.innovationID):
                selectedLink = random.choice([currentMum, currentDad])

                mumIt += 1
                dadIt += 1

            elif (currentMum.innovationID < currentDad.innovationID):
                if (best == mum):
                    selectedLink = currentMum

                mumIt += 1

            elif (currentDad.innovationID < currentMum.innovationID):
                if (best == dad):
                    selectedLink = currentDad

                dadIt += 1

            if (selectedLink != None):
                if (len(babyLinks) == 0):
                    babyLinks.append(selectedLink)
                else:
                    if (babyLinks[-1].innovationID != selectedLink.innovationID):
                        babyLinks.append(selectedLink)

                if (selectedLink.fromNeuron not in babyNeurons):
                    babyNeurons.append(selectedLink.fromNeuron)

                if (selectedLink.toNeuron not in babyNeurons):
                    babyNeurons.append(selectedLink.toNeuron)

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        # self.currentGenomeID += 1

        if (len(babyLinks) < 1 or len(babyNeurons) < 1):
            best = random.choice([mum, dad])
            babyNeurons = best.neurons
            babyLinks = best.links

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, mum.inputs, mum.outputs)
