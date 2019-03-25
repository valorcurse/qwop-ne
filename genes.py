from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any

import random
from random import randint

import math
from math import cos, sin, atan, ceil, floor
from sklearn.preprocessing import normalize
from enum import Enum
import itertools

import numpy as np
from matplotlib import pyplot
import matplotlib.patches as patches

from prettytable import PrettyTable

from phenotypes import CNeuralNet, SLink, SNeuron, NeuronType
# from neat import CSpecies

class CSpecies:
    pass

class Phase(Enum):
    COMPLEXIFYING = 0
    PRUNING = 1

class SpeciationType(Enum):
    COMPATIBILITY_DISTANCE = 0
    NOVELTY = 1

class MutationRates:
    def __init__(self) -> None:
        self.crossoverRate = 0.7

        self.newSpeciesTolerance = 3.0

        self.chanceToMutateBias = 0.7

        self.chanceToAddNode = 0.15
        # self.chanceToAddNode = 0.25
        self.chanceToAddLink = 0.3
        # self.chanceToAddLink = 0.5

        self.chanceToAddRecurrentLink = 0.05

        self.chanceToDeleteNeuron = 0.15
        self.chanceToDeleteLink = 0.3

        self.mutationRate = 0.5
        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.mpcMargin = 20

class InnovationType(Enum):
    NEURON = 0
    LINK = 1

class SInnovation:
    def __init__(self, innovationType: InnovationType, innovationID: int, start: Optional[int], end: Optional[int], neuronID: Optional[int]) -> None:
        self.innovationType = innovationType
        self.innovationID = innovationID
        self.start = start
        self.end = end
        self.neuronID = neuronID

    def __eq__(self, other: object) -> bool:
        return self.innovationType == other if isinstance(other, SInnovation) else NotImplemented

class SNeuronGene:
    def __init__(self, neuronType: NeuronType, ID: int, y: float, innovationID: int) -> None:
        self.neuronType = neuronType
        self.ID = ID
        self.splitY = y
        self.innovationID = innovationID
        
        self.bias = 0.0
        self.recurrent = False

    def __eq__(self, other: Any) -> bool:
        return other is not None and self.innovationID == other.innovationID

class SLinkGene:

    def __init__(self, fromNeuron: SNeuronGene, toNeuron: SNeuronGene, enabled: bool, innovationID: int, weight: float, recurrent: bool=False):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.enabled = enabled

        self.recurrent = recurrent

        self.innovationID = innovationID

    def __lt__(self, other: SLinkGene) -> bool:
        return self.innovationID < other.innovationID

    def __eq__(self, other: object) -> bool:
        return self.innovationID == other.innovationID if isinstance(other, SLinkGene) else NotImplemented

class Innovations:
    def __init__(self) -> None:
        self.listOfInnovations: List[SInnovation] = []

        self.currentNeuronID = 1

    def createNewLinkInnovation(self, fromID: int, toID: int) -> int:
        ID: int = innovations.checkInnovation(fromID, toID, InnovationType.LINK)

        if (ID == -1):
            newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromID, toID, None)
            self.listOfInnovations.append(newInnovation)
            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewLink(self, fromNeuron: SNeuronGene, toNeuron: SNeuronGene, enabled: bool, weight: float, recurrent: bool=False) -> SLinkGene:
        ID = self.createNewLinkInnovation(fromNeuron.ID, toNeuron.ID)

        return SLinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

    def createNewNeuronInnovation(self, fromID: Optional[int], toID: Optional[int], neuronID: int) -> int:
        ID: int = innovations.checkInnovation(
            fromID, 
            toID, 
            InnovationType.NEURON,
            neuronID)
        
        if (ID == -1):
            newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations),
                                        fromID, toID, neuronID)

            self.listOfInnovations.append(newInnovation)

            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewNeuron(self, y: float, neuronType: NeuronType, fromNeuron: Optional[SNeuronGene] = None, toNeuron: Optional[SNeuronGene] = None, neuronID: Optional[int] = None) -> SNeuronGene:
        if (neuronID is None):
            neuronID = self.currentNeuronID
            self.currentNeuronID += 1
        
        fromID = fromNeuron.ID if fromNeuron else None
        toID = toNeuron.ID if toNeuron else None

        innovationID = self.createNewNeuronInnovation(fromID, toID, neuronID)

        return SNeuronGene(neuronType, neuronID, y, innovationID)
    
    def checkInnovation(self, start: Optional[int], end: Optional[int], innovationType: InnovationType, neuronID: Optional[int] = None) -> int:
        matchingInnovations = [innovation for innovation in self.listOfInnovations 
                if innovation.start == start 
                and innovation.end == end
                and innovation.neuronID == neuronID
                and innovation.innovationType == innovationType]


        return matchingInnovations[0].innovationID if len(matchingInnovations) > 0 else -1

    def printTable(self) -> None:
        table = PrettyTable(["ID", "type", "start", "end", "neuron ID"])
        for innovation in self.listOfInnovations:
            if (innovation.innovationType == InnovationType.NEURON and innovation.neuronID is not None and innovation.neuronID < 0):
                continue

            table.add_row([
                innovation.innovationID,
                innovation.innovationType, 
                innovation.start if innovation.start else "None", 
                innovation.end if innovation.end else "None", 
                innovation.neuronID])

        print(table)



# Global innovations database
global innovations
innovations = Innovations()

class CGenome:

    def __init__(self, ID: int, neurons: List[SNeuronGene], links: List[SLinkGene], inputs: int, outputs: int, parents: List[CGenome]=[]) -> None:
        self.ID = ID
        self.parents = parents

        self.inputs = inputs
        self.outputs = outputs
        
        self.links = links
        self.neurons = neurons
        
        if (len(self.neurons) == 0):
            for n in range(inputs):
                newNeuron = innovations.createNewNeuron(0.0, NeuronType.INPUT, neuronID = -n-1)
                self.neurons.append(newNeuron)

            for n in range(outputs):
                newNeuron = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, neuronID = -inputs-n-1)
                self.neurons.append(newNeuron)

        self.fitness: float = 0.0
        self.adjustedFitness: float = 0.0
        self.novelty: float = 0.0
        
        self.milestone: float = 0.0

        # For printing
        self.distance: float = 0.0
        
        self.species: Optional[CSpecies] = None            

    def __lt__(self, other: CGenome) -> bool:
        return self.fitness < other.fitness

    def getLinksIn(self, neuron: SNeuronGene) -> List[SLinkGene]:
        return [l for l in self.links if l.toNeuron == neuron]

    def getLinksOut(self, neuron: SNeuronGene) -> List[SLinkGene]:
        return [l for l in self.links if l.fromNeuron == neuron]

    def isNeuronValid(self, neuron: SNeuronGene) -> bool:
        return len(self.getLinksIn(neuron)) >= 1 or len(self.getLinksOut(neuron)) >= 1

    def calculateCompatibilityDistance(self, other: CGenome) -> float:
        disjointRate = 1.0
        matchedRate = 0.4

        disjointedLinks = 0.0
        weightDifferences = []

        combinedIndexes = list(set(
            [l.innovationID for l in self.links] + [l.innovationID for l in other.links]))
        combinedIndexes.sort()
        
        selfLinksDict = {l.innovationID: l for l in self.links}
        otherLinksDict = {l.innovationID: l for l in other.links}

        for i in combinedIndexes:
            selfLink = selfLinksDict.get(i)
            otherLink = otherLinksDict.get(i)



            if ((selfLink is None) or (otherLink is None)):
                disjointedLinks += 1.0
            else:
                weightDifferences.append(math.fabs(selfLink.weight - otherLink.weight))
        
        longestLinks = max(1.0, max(len(other.links), len(self.links)))
        # longestLinks = 1.0 if longestLinks <= 20 else longestLinks
        # weightDifference = 0.0 if len(weightDifferences) == 0 else np.mean(weightDifferences)
        weightDifference = 0.0 if len(weightDifferences) == 0 else np.sum(weightDifferences)

        linkDistance = (disjointRate * disjointedLinks / longestLinks) + weightDifference * matchedRate
        # print(linkDistance)

        disjointedNeurons = 0.0
        biasDifferences = []

        combinedNeurons = list(set(
            [n.innovationID for n in self.neurons if n.neuronType == NeuronType.HIDDEN] + 
            [n.innovationID for n in other.neurons if n.neuronType == NeuronType.HIDDEN]))
        combinedNeurons.sort()
        
        selfNeuronsDict = {n.innovationID: n for n in self.neurons}
        otherNeuronDict = {n.innovationID: n for n in other.neurons}


        for i in combinedNeurons:
            selfNeuron = selfNeuronsDict.get(i)
            otherNeuron = otherNeuronDict.get(i)


                # otherNeuron.innovationID if otherNeuron else "None"))

            if (selfNeuron is None or otherNeuron is None):
                disjointedNeurons += 1.0
            else:
                biasDifferences.append(math.fabs(selfNeuron.bias - otherNeuron.bias))

        longestNeurons = max(1.0, max(len(other.neurons), len(self.neurons)))
        # longestNeurons = 1.0 if longestNeurons <= 20 else longestNeurons
        biasDifference = 0.0 if len(biasDifferences) == 0 else np.mean(biasDifferences)

        neuronDistance = (disjointRate * disjointedNeurons / longestNeurons) + biasDifference * matchedRate

        distance: float = linkDistance + neuronDistance
        self.distance = distance
        
        return distance
        # return linkDistance

    def addRandomLink(self, chanceOfLooped: float) -> None:

        fromNeuron: Optional[SNeuronGene] = None
        toNeuron: Optional[SNeuronGene] = None
        recurrent: bool = False

        # Add recurrent link
        # if (random.random() < chanceOfLooped and len(self.neurons) > (self.inputs + self.outputs)):
        #     possibleNeurons = [n for n in self.neurons
        #         if not n.recurrent and n.neuronType == NeuronType.HIDDEN]

        #     if (len(possibleNeurons) == 0):
        #         return

        #     loopNeuron = random.choice(possibleNeurons)
        #     fromNeuron = toNeuron = loopNeuron
        #     recurrent = loopNeuron.recurrent = True

        # else:

        keepLoopRunning = True
        fromNeurons = [neuron for neuron in self.neurons
                       if (neuron.neuronType in [NeuronType.INPUT, NeuronType.HIDDEN])]

        toNeurons = [neuron for neuron in self.neurons
                     if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
        
        triesToAddLink = 10  
        while (triesToAddLink > 0):
            
            fromNeuron = random.choice(fromNeurons)
            toNeuron = random.choice(toNeurons)

            # If link already exists
            alreadyExists = next(
                (l for l in self.links if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID)), 
                None)

            if (not alreadyExists and fromNeuron.ID != toNeuron.ID and fromNeuron.splitY < toNeuron.splitY):
                break
            else:
                fromNeuron = toNeuron = None

            triesToAddLink -= 1

        if (fromNeuron is None or toNeuron is None):
            return

        # if (fromNeuron.splitY > toNeuron.splitY):
            # recurrent = True

        self.addLink(fromNeuron, toNeuron, recurrent)
    
    def addLink(self, fromNeuron: SNeuronGene, toNeuron: SNeuronGene, weight: float = 1.0, recurrent: bool = False) -> None:
        link = innovations.createNewLink(fromNeuron, toNeuron, True, 1.0, recurrent)
        self.links.append(link)


    def removeRandomLink(self) -> None:
        if (len(self.links) == 0):
            return

        randomLink = random.choice(self.links)

        self.removeLink(randomLink)

    def removeLink(self, link: SLinkGene) -> None:

        fromNeuron = link.fromNeuron

        if fromNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(fromNeuron):
            self.removeNeuron(fromNeuron)


        toNeuron = link.toNeuron
        if toNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(toNeuron):
            self.removeNeuron(toNeuron)

        self.links.remove(link)

    def addNeuron(self) -> None:

        if (len(self.links) < 1):
            return

        maxRand = len(self.links)

        possibleLinks = [l for l in self.links[:maxRand] if l.enabled]

        if (len(possibleLinks) == 0):
            return

        chosenLink = random.choice(possibleLinks)
        

        originalWeight = chosenLink.weight
        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newDepth = (fromNeuron.splitY + toNeuron.splitY) / 2

        newNeuron = innovations.createNewNeuron(newDepth, NeuronType.HIDDEN, fromNeuron, toNeuron)

        self.addLink(fromNeuron, newNeuron)
        self.addLink(newNeuron, toNeuron)

        self.removeLink(chosenLink)

        self.neurons.append(newNeuron)
        self.neurons.sort(key=lambda x: x.splitY, reverse=False)


    def removeRandomNeuron(self) -> None:
        # Get all the hidden neurons which do not have multiple incoming AND outgoing links

        possibleNeurons = [n for n in self.neurons if n.neuronType == NeuronType.HIDDEN 
            and ((len(self.getLinksOut(n)) == 1 and len(self.getLinksIn(n)) >= 1)
            or (len(self.getLinksOut(n)) >= 1 and len(self.getLinksIn(n)) == 1))]

        if (len(possibleNeurons) == 0):
            return

        randomNeuron = random.choice(possibleNeurons)
        self.removeNeuron(randomNeuron)

    def removeNeuron(self, neuron: SNeuronGene) -> None:
        linksIn = self.getLinksIn(neuron)
        linksOut = self.getLinksOut(neuron)
        if len(linksIn) > 1 and len(linksOut) > 1:
            return

        if len(linksOut) == 1:
            patchThroughNeuron = linksOut[0].toNeuron


            for link in linksIn:
                originNeuron = link.fromNeuron

                self.removeLink(link)
                self.addLink(originNeuron, patchThroughNeuron, link.weight)

        elif len(linksIn) == 1:
            originNeuron = linksIn[0].fromNeuron

            for link in linksIn:
                patchThroughNeuron = link.toNeuron

                self.removeLink(link)
                self.addLink(originNeuron, patchThroughNeuron, link.weight)

        self.neurons.remove(neuron)


    def mutateWeights(self, mutationRate: float, replacementProbability: float, maxWeightPerturbation: float) -> None:
            for link in self.links:
                if (random.random() > 0.25):
                    continue

                if (random.random() < mutationRate):
                    link.weight += random.gauss(0.0, maxWeightPerturbation)
                    link.weight = min(30.0, max(-30.0, link.weight))

                # elif (random.random() < replacementProbability):
                else:
                    link.weight = random.gauss(0.0, maxWeightPerturbation)


    def mutateBias(self, mutationRate: float, replacementProbability: float, maxWeightPerturbation: float) -> None:
        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() < mutationRate):
                n.bias += random.gauss(0.0, maxWeightPerturbation)
                n.bias = min(30.0, max(-30.0, n.bias))

            elif (random.random() < replacementProbability):
                n.bias = random.gauss(0.0, maxWeightPerturbation)

    def mutate(self, phase: Phase, mutationRates: MutationRates) -> None:
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


        # if phase == Phase.COMPLEXIFYING:
        if (random.random() < mutationRates.chanceToAddNode):
            self.addNeuron()

        if (random.random() < mutationRates.chanceToAddLink):
            self.addRandomLink(mutationRates.chanceToAddRecurrentLink)

        # elif phase == Phase.PRUNING:
        if (random.random() < mutationRates.chanceToAddNode):
            self.removeRandomNeuron()

        if (random.random() < mutationRates.chanceToAddLink):
            self.removeRandomLink()

        self.mutateWeights(mutationRates.mutationRate, mutationRates.probabilityOfWeightReplaced, mutationRates.maxWeightPerturbation)
        self.mutateBias(mutationRates.chanceToMutateBias, mutationRates.probabilityOfWeightReplaced, mutationRates.maxWeightPerturbation)

        self.links.sort()

    def createPhenotype(self) -> CNeuralNet:
        phenotypeNeurons = []

        for neuron in self.neurons:
            newNeuron = SNeuron(neuron.neuronType,
                        neuron.ID,
                        neuron.bias,
                        neuron.splitY)

            phenotypeNeurons.append(newNeuron)



        for link in self.links:
            if (link.enabled):
                fromNeuron = next((neuron
                                   for neuron in phenotypeNeurons if (neuron.ID == link.fromNeuron.ID)), None)
                toNeuron = next((neuron
                                 for neuron in phenotypeNeurons if (neuron.ID == link.toNeuron.ID)), None)

                if (not fromNeuron) or (not toNeuron):
                    continue

                tmpLink = SLink(fromNeuron,
                                toNeuron,
                                link.weight,
                                link.recurrent)

                toNeuron.linksIn.append(tmpLink)

        return CNeuralNet(phenotypeNeurons, self.ID)
