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
from genes import MutationRates
from genes import Phase
from genes import SpeciationType

global innovations

class CSpecies:
    numGensAllowNoImprovement = 40

    def __init__(self, speciesID: int, leader: CGenome):
        self.ID: int = speciesID

        self.members: CGenome = [leader]

        self.age: int = 0
        self.numToSpawn: int = 0

        self.leader: CGenome = leader

        self.youngAgeThreshold: int = 10
        self.youngAgeBonus: float = 1.5
        self.oldAgeThreshold: int = 50
        self.oldAgePenalty: float = 0.5

        self.highestFitness: float = 0.0
        self.generationsWithoutImprovement: int = 0

        self.stagnant = False

    def __contains__(self, key: int):
        return key.ID in [m.ID for m in self.members]

    def isMember(self, genome: CGenome):
        # print("members:", [m.ID for m in self.members], "genome:", genome.ID, "->", (genome.ID in [m.ID for m in self.members]))
        return (genome.ID in [m.ID for m in self.members])

    def best(self):
        # return max(self.members, key=attrgetter('fitness'))
        return max(self.members)

    def spawn(self):
        return random.choice(self.members)

    def adjustFitnesses(self):
        # fitnessRange = max(1.0, maxFitness - minFitness)
        
        # highestFitness = round(highestFitness, 1)

        # avgMemberFitness = sum([m.fitness for m in self.members])/len(self.members)
        # newAdjustedFitness = (avgMemberFitness - minFitness) / fitnessRange

        for m in self.members:
            m.adjustedFitness = m.fitness / len(self.members)

            if self.age <= self.youngAgeThreshold:
                m.adjustedFitness *= self.youngAgeBonus

            if self.age >= self.oldAgeThreshold:
                m.adjustedFitness *= self.oldAgePenalty

        # self.adjustedFitness = newAdjustedFitness

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

    def __init__(self, numberOfGenomes, numOfInputs, numOfOutputs, mutationRates=MutationRates(), fullyConnected=False):

        self.genomes: CGenome = []
        self.phenotypes: CNeuralNet = []
        self.species: CSpecies = []
        self.speciesNumber: int = 0

        self.populationSize: int = numberOfGenomes

        self.generation: int = 0

        self.currentGenomeID: int = 0

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.mutationRates: MutationRates = mutationRates

        inputs = []
        for n in range(numOfInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(numOfInputs) + ")", end='')
            newInput = innovations.createNewNeuron(0.0, None, None, NeuronType.INPUT, -n-1)
            inputs.append(newInput)

        print("")

        outputs = []
        for n in range(numOfOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(numOfOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(1.0, None, None, NeuronType.OUTPUT, -numOfInputs-n-1)
            outputs.append(newOutput)

        print("")

        links = []
        if fullyConnected:
            for input in inputs:
                for output in outputs:
                    new_link = innovations.createNewLink(input, output, True, 1.0)
                    links.append(new_link)

        inputs.extend(outputs)
        newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
        for i in range(self.populationSize):
            print("\rCreating genomes (" + str(i + 1) + "/" + str(self.populationSize) + ")", end='')
            self.genomes.append(deepcopy(newGenome))
            self.currentGenomeID += 1

        print("")


        # mpc = self.calculateMPC()
        mpc = 100
        self.mpcThreshold: int = mpc + mutationRates.mpcMargin
        self.lowestMPC: int = mpc
        self.mpcStagnation: int = 0
        
        print("mpc", mpc)
        print("mpc threshold", self.mpcThreshold)

        self.speciate()
        self.epoch([0]*len(self.genomes))


    def calculateMPC(self):
        allMutations = [[n for n in g.neurons] + [l for l in g.links] for g in self.genomes]
        nrOfMutations = len([item for sublist in allMutations for item in sublist])
        return (nrOfMutations / len(self.genomes))

    def epoch(self, fitnessScores: float, novelty: float = None):
        
        if novelty is not None:
            for index, genome in enumerate(self.genomes):
                genome.novelty = novelty[index]

        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = max(1, fitnessScores[index])

        self.calculateSpawnAmount()
        self.reproduce()
        
        self.speciate()

        mpc = self.calculateMPC()

        if self.phase == Phase.PRUNING:
            if mpc < self.lowestMPC:
                self.lowestMPC = mpc
                self.mpcStagnation = 0
            else:
                self.mpcStagnation += 1

            if self.mpcStagnation >= 10:
                self.phase = Phase.COMPLEXIFYING
                self.mpcThreshold = mpc + self.mutationRates.mpcMargin

        elif self.phase == Phase.COMPLEXIFYING:
            if mpc >= self.mpcThreshold:
                self.phase = Phase.PRUNING
                self.mpcStagnation = 0
                self.lowestMPC = mpc


        newPhenotypes = []
        for genome in self.genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.generation += 1

        self.phenotypes = newPhenotypes

    def speciate(self):
        # Find best leader for species from the new population
        unspeciated = list(range(0, len(self.genomes)))
        for s in self.species:
            compareMember = random.choice(s.members)
            s.members = []
            candidates = []
            for i in unspeciated:
                genome = self.genomes[i]
                distance = genome.calculateCompatibilityDistance(compareMember)
                if (distance < self.mutationRates.newSpeciesTolerance):
                    candidates.append((distance, i))

            _, bestCandidate = min(candidates, key=lambda x: x[0])

            s.leader = self.genomes[bestCandidate]
            s.members.append(s.leader)
            unspeciated.remove(bestCandidate)

        # Distribute genomes to their closest species
        for i in unspeciated:
            genome = self.genomes[i]

            closestDistance = self.mutationRates.newSpeciesTolerance
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
                # random: float = random.random()

                # if (random <= 0.1):
                self.speciesNumber += 1
                self.species.append(CSpecies(self.speciesNumber, genome))
                # else:
                    


    def reproduce(self):
        newPop = []
        for s in self.species:
            numToSpawn = s.numToSpawn

            members = deepcopy(s.members)
            members.sort(reverse=True, key=lambda x: x.fitness)

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

            for i in range(numToSpawn):
                baby = None

                if (self.phase == Phase.PRUNING or random.random() > self.mutationRates.crossoverRate):
                    baby = deepcopy(random.choice(members))
                else:
                    # g1 = random.choice(members)
                    # g2 = random.choice(members)
                    
                    # Tournament selection
                    g1 = sorted(random.sample(members, 5), key=lambda x: x.fitness)[0]
                    g2 = sorted(random.sample(members, 5), key=lambda x: x.fitness)[0]
                    
                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
                # baby.mutate(self.phase, self.mutationRates)

                newPop.append(baby)

        self.genomes = newPop

    def calculateSpawnAmount(self):
        # Remove stagnant species
        if self.phase == Phase.COMPLEXIFYING:
            for s in self.species:
                s.becomeOlder()

                if s.stagnant:
                    self.species.remove(s)
        
        # Adjust species fitness
        # allFitnesses = [m.fitness for spc in self.species for m in spc.members]
        # minFitness = min(allFitnesses) if len(allFitnesses) != 0 else 0.0
        # maxFitness = max(allFitnesses) if len(allFitnesses) != 0 else 0.0
        for s in self.species:
            s.adjustFitnesses()
        
        allFitnesses = sum([m.adjustedFitness for spc in self.species for m in spc.members])
        


        print("allFitnesses", allFitnesses)
        for s in self.species:
            sumOfFitnesses = sum([m.adjustedFitness for m in s.members])


            portionOfFitness = 1 if allFitnesses == 0 and sumOfFitnesses == 0 else sumOfFitnesses/allFitnesses
            s.numToSpawn = int(portionOfFitness * self.populationSize)
        
        # minToSpawn = 2
        # sumFitness = sum(s.adjustedFitness for s in self.species)
        # spawnAmounts = []
        
        # for s in self.species:            
        #     if (sumFitness > 0):
        #         size = max(minToSpawn, s.adjustedFitness / sumFitness * self.populationSize)
        #     else:
        #         size = minToSpawn

        #     previousSize = len(s.members)
        #     sizeDifference = (size - previousSize) * 0.5
        #     roundedSize = int(round(sizeDifference))
        #     toSpawn = previousSize
            
        #     if abs(roundedSize) > 0:
        #         toSpawn += roundedSize
        #     elif sizeDifference > 0:
        #         toSpawn += 1
        #     elif sizeDifference < 0:
        #         toSpawn -= 1

        #     spawnAmounts.append(toSpawn)

        # totalSpawn = max(1, sum(spawnAmounts))
        # norm = self.populationSize / totalSpawn
        # spawnAmounts = [max(minToSpawn, int(round(n * norm))) for n in spawnAmounts]
        
        # for spawnAmount, species in zip(spawnAmounts, self.species):
        #     species.numToSpawn = spawnAmount

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
            
            if (mumLink is None):
                if (best == dad):
                    babyLinks.append(dadLink)

            elif (dadLink is None):
                if (best == mum):
                    babyLinks.append(mumLink)

            else:
                babyLinks.append(random.choice([mumLink, dadLink]))

        for link in babyLinks:

            if (link.fromNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(deepcopy(link.fromNeuron))

            if (link.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(deepcopy(link.toNeuron))

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, best.inputs, best.outputs)

