import random
from random import randint

import math
from enum import Enum


class InnovationType(Enum):
    NEURON = 0
    LINK = 1


class NeuronType(Enum):
    INPUT = 0
    HIDDEN = 1
    BIAS = 2
    OUTPUT = 3
    LINK = 4


class SInnovation:
    def __init__(self, innovationType, innovationID, start, end, neuronID, neuronType):
        self.innovationType = innovationType
        self.innovationID = innovationID
        self.start = start
        self.end = end
        self.neuronID = neuronID
        self.neuronType = neuronType


class Innovations:
    listOfInnovations = []

    currentNeuronID = 0

    def createNewLinkInnovation(self, fromNeuron, toNeuron):
        newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromNeuron, toNeuron, -1,
                                    NeuronType.LINK)
        self.listOfInnovations.append(newInnovation)

        return len(self.listOfInnovations) - 1;

    def createNewLink(self, fromNeuron, toNeuron, enabled, weight, recurrent=False):
        ID = self.createNewLinkInnovation(fromNeuron, toNeuron)
        return SLinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

    def createNewNeuronInnovation(self, fromNeuron, toNeuron):
        neurons = [neuron for neuron in self.listOfInnovations if neuron.innovationType == InnovationType.NEURON]
        newNeuronID = len(neurons)
        newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations),
                                    fromNeuron, toNeuron, newNeuronID, NeuronType.HIDDEN)
        self.listOfInnovations.append(newInnovation)

        return len(self.listOfInnovations) - 1;

    def createNewNeuron(self, fromNeuron, toNeuron, x, y, neuronType, recurrent=False):
        ID = self.createNewNeuronInnovation(fromNeuron, toNeuron)
        self.currentNeuronID += 1
        return SNeuronGene(neuronType, self.currentNeuronID, x, y, ID, recurrent)

    def checkInnovation(self, start, end, innovationType):
        matched = next((innovation for innovation in self.listOfInnovations if (
                (innovation.start == start) and
                (innovation.end == end) and
                (innovation.innovationType == innovationType))), None)
        # print(matched)
        return -1 if (matched == None) else matched.innovationID

    def getInnovation(self, innovationID):
        return self.listOfInnovations[innovationID]


# Global innovations database
global innovations
innovations = Innovations()


class SLinkGene:

    def __init__(self, fromNeuron, toNeuron, enabled, innovationID, weight, recurrent=False):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.enabled = enabled

        self.recurrent = recurrent

        self.innovationID = innovationID

    def __lt__(self, other):
        return self.innovationID < other.innovationID


class SNeuronGene:
    def __init__(self, neuronType, ID, x, y, innovationID, recurrent=False):
        self.ID = ID
        self.neuronType = neuronType
        self.recurrent = recurrent
        self.activationResponse = None
        self.splitX = x
        self.splitY = y

        self.innovationID = innovationID

        self.numInputs = 0
        self.numOutputs = 0

        self.neurons = []
        self.links = []


class SLink:

    def __init__(self, neuronIn, neuronOut, weight, recurrent=False):
        self.neuronIn = neuronIn
        self.neuronOut = neuronOut

        self.weight = weight

        self.recurrent = recurrent


class SNeuron:
    def __init__(self, neuronType, neuronID, y, x, activationResponse):
        self.linksIn = []
        self.linksOut = []

        self.sumActivation = 0.0
        self.output = 0.0

        self.neuronType = neuronType

        self.ID = neuronID

        self.activationResponse = activationResponse

        self.posX = self.posY = 0
        self.splitX = x
        self.splitY = y


class CGenome:

    def __init__(self, ID, neurons, links, inputs, outputs):
        self.ID = ID
        self.neurons = neurons
        self.links = links
        self.inputs = inputs
        self.outputs = outputs
        
        self.adjustedFitness = 0.0
        self.fitness = 0.0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def calculateCompatibilityDistance(self, otherGenome):
        g1 = g2 = 0
        numExcess = 0.0
        numMatched = 1.0
        numDisjointed = 0.0

        weightDifference = 0.0

        # if (len(self.links) < 1 or len(otherGenome.links) < 1):
        # return 0

        while ((g1 < len(self.links) - 1) or (g2 < len(otherGenome.links) - 1)):

            if (g1 == len(self.links) - 1):
                g2 += 1
                numExcess += 1

                continue

            if (g2 == len(otherGenome.links) - 1):
                g1 += 1
                numExcess += 1

                continue

            id1 = self.links[g1].innovationID
            id2 = otherGenome.links[g2].innovationID

            if (id1 == id2):
                g1 += 1
                g2 += 1
                numMatched += 1

                weightDifference += math.fabs(self.links[g1].weight - otherGenome.links[g2].weight)

            if (id1 < id2):
                numDisjointed += 1
                g1 += 1

            if (id1 > id2):
                numDisjointed += 1
                g2 += 1

        # longest = len(otherGenome.links) if len(otherGenome.links) > len(self.links) else len(self.links)
        longest = max(len(otherGenome.links), len(self.links))
        longest = 1.0 if longest < 20 else longest
        # longest = 1.0

        # print("links: ", len(self.links), len(otherGenome.links))
        # print("longest: " + str(longest))

        disjoint = 1.0
        excess = 1.0
        matched = 0.4

        # print([[l.fromNeuron.ID, l.toNeuron.ID] for l in self.links])
        # print([[l.fromNeuron.ID, l.toNeuron.ID] for l in otherGenome.links])
        # print("Disjoint:", numDisjointed, "Excess:", numExcess, "Weight diff:", weightDifference)

        return ((excess * numExcess / longest) +
                (disjoint * numDisjointed / longest) +
                (matched * weightDifference / numMatched))

    def addLink(self, mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink):

        if (random.random() > mutationRate):
            return

        neuron1 = None
        neuron2 = None
        recurrent = False

        if (random.random() < chanceOfLooped):
            loopFound = False
            while (triesToFindLoop and not loopFound):
                neuronPosition = randint(self.inputs + 1, len(self.neurons) - 1)

                loopNeuron = self.neurons[neuronPosition]
                if (not loopNeuron.recurrent or
                        loopNeuron.neuronType != NeuronType.BIAS or
                        loopNeuron.neuronType != NeuronType.INPUT):
                    neuron1 = neuron2 = loopNeuron

                    recurrent = loopNeuron.recurrent = True

                    loopFound = True
        else:
            loopFound = False
            while (triesToAddLink and not loopFound):
                fromNeurons = [neuron for neuron in self.neurons
                               if (neuron.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.HIDDEN])]
                neuron1 = random.choice(fromNeurons)

                toNeurons = [neuron for neuron in self.neurons
                             if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
                neuron2 = random.choice(toNeurons)

                # TODO: why?
                # if (neuron2.ID is 2):
                # continue

                linkIsDuplicate = next(
                    (l for l in self.links
                     if (l.fromNeuron == neuron1) and
                     (l.toNeuron == neuron2)),
                    None)

                if (not linkIsDuplicate or neuron1.ID == neuron2.ID):
                    loopFound = True
                else:
                    neuron1 = neuron2 = None

                triesToAddLink -= 1

        if (neuron1 == None or neuron2 == None):
            return

        # print("Creating link:", neuron1.neuronType, neuron2.neuronType)

        ID = innovations.checkInnovation(neuron1, neuron2, InnovationType.LINK)

        if (neuron1.splitY > neuron2.splitY):
            recurrent = True

        randomClamped = random.random() - random.random()
        if (ID < 0):
            # print("Adding new link")
            ID = innovations.createNewLinkInnovation(neuron1, neuron2)

            newGene = SLinkGene(neuron1, neuron2, True, ID, randomClamped, recurrent)
            self.links.append(newGene)

        else:
            newGene = SLinkGene(neuron1, neuron2, True, ID, randomClamped, recurrent)
            self.links.append(newGene)

    def addNeuron(self, mutationRate, triesToFindOldLink):

        if (len(self.links) < 5):
            return

        if (random.random() > mutationRate):
            return

        done = False
        chosenLink = 0

        sizeThreshold = self.inputs + self.outputs + 5
        if (len(self.neurons) < sizeThreshold):

            loopFound = False
            while (triesToFindOldLink and not loopFound):
                maxRand = math.floor(len(self.links) - 1 - math.sqrt(len(self.links)))
                chosenLink = self.links[randint(0, maxRand)]

                fromNeuron = chosenLink.fromNeuron

                if (chosenLink.enabled and
                        not chosenLink.recurrent and
                        fromNeuron.neuronType != NeuronType.BIAS):
                    done = loopFound = True

                triesToFindOldLink -= 1

            if (not done):
                return

        else:

            while (not done):
                chosenLink = self.links[randint(0, len(self.links) - 1)]
                fromNeuron = chosenLink.fromNeuron

                if (chosenLink.enabled and (not chosenLink.recurrent) and (fromNeuron.neuronType != NeuronType.BIAS)):
                    done = True

        chosenLink.enabled = False

        originalWeight = chosenLink.weight

        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newDepth = (fromNeuron.splitY + toNeuron.splitY) / 2
        newWidth = (fromNeuron.splitX + toNeuron.splitX) / 2

        ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.NEURON)

        if (ID >= 0):
            neuron = innovations.getInnovation(ID)

            if (neuron != None):
                ID = -1

        if (ID < 0):
            # print("Adding new neuron.")
            newNeuron = innovations.createNewNeuron(fromNeuron, toNeuron, newWidth, newDepth, NeuronType.HIDDEN)
            self.neurons.append(newNeuron)

            link1 = innovations.createNewLink(fromNeuron, newNeuron, True, 1.0)
            link2 = innovations.createNewLink(newNeuron, toNeuron, True, originalWeight)

            self.links.append(link1)
            self.links.append(link2)

        else:

            newNeuron = innovations.getInnovation(ID)

            idLink1 = innovations.checkInnovation(fromNeuron, newNeuron, NeuronType.LINK)
            idLink2 = innovations.checkInnovation(newNeuron, toNeuron, NeuronType.LINK)

            if (idLink1 < 0 or idLink2 < 0):
                return

            link1 = SLinkGene(fromNeuron, newNeuron, True, idLink1, 1.0)
            link2 = SLinkGene(newNeuron, toNeuron, True, idLink2, originalWeight)

            self.links.append(link1)
            self.links.append(link2)

            newNeuron = SNeuronGene(NeuronType.HIDDEN, newNeuron, newWidth, newDepth)

            self.neurons.append(newNeuron)

        self.neurons.sort(key=lambda x: x.splitY, reverse=False)

    def mutateWeights(self, mutationRate, replacementProbability, maxWeightPerturbation):
        if (random.random() < mutationRate):
            # print("Mutating weights")
            for link in self.links:
                if (random.random() < replacementProbability):
                    link.weight = random.random()
                    return

                # if (random.random() < mutationRate):
                else:
                    link.weight += random.uniform(-1, 1) * maxWeightPerturbation

    def createPhenotype(self, depth):
        # deletePhenotype()

        phenotypeNeurons = []

        # print("creating phenotype:")
        for neuron in self.neurons:
            # print("neuron id:", neuron.ID)
            phenotypeNeurons.append(
                SNeuron(neuron.neuronType,
                        neuron.ID,
                        neuron.splitY,
                        neuron.splitX,
                        neuron.activationResponse))

        for link in self.links:
            if (link.enabled):
                # print("link:", link)
                fromNeuron = next((neuron
                                   for neuron in (phenotypeNeurons) if (neuron.ID == link.fromNeuron.ID)), None)
                toNeuron = next((neuron
                                 for neuron in (phenotypeNeurons) if (neuron.ID == link.toNeuron.ID)), None)

                tmpLink = SLink(fromNeuron,
                                toNeuron,
                                link.weight,
                                link.recurrent)

                fromNeuron.linksOut.append(tmpLink)
                toNeuron.linksIn.append(tmpLink)

        return CNeuralNet(phenotypeNeurons, depth, self.ID)


class CNeuralNet:

    def __init__(self, neurons, depth, ID):
        self.neurons = neurons
        self.depth = depth
        self.ID = ID

    # print("--------------------------------------------------------------------------------")
    # print("neural network neurons:", len(self.neurons))
    # for neuron in self.neurons:
    # 	print(neuron.neuronType, neuron.ID, neuron.splitY)
    # 	print("\tlinksin:", [link.neuronIn.ID for link in neuron.linksIn])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def update(self, inputs):

        # if (len(self.neurons) < 1 or len(self.links) < 1):
        # return [0]*len([neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT])

        outputs = []

        neuronIndex = 0
        # print("inputs", inputs)

        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for inputNeuron in inputs:
            # print("inputNeuron:", inputNeuron)
            # print("index:", neuronIndex)
            self.neurons[neuronIndex].output = inputNeuron
            neuronIndex += 1

        # Set bias
        self.neurons[neuronIndex].output = 1

        # neuronIndex += 1

        # neuronIndex = 0
        # print("--------------------------------------------------------------------------------")
        for currentNeuron in self.neurons:
            neuronSum = 0.0

            # print("neuron:", currentNeuron.ID, currentNeuron.neuronType, currentNeuron.splitY)
            for link in currentNeuron.linksIn:
                weight = link.weight

                # print(link.neuronIn.output, weight)
                # print("neuronin: ", link.neuronIn, link.neuronIn.output)
                neuronOutput = link.neuronIn.output
                # print("neuronOutput:", neuronOutput)
                # print("neuronSum=", weight, neuronOutput, (weight * neuronOutput))
                neuronSum += weight * neuronOutput

            # currentNeuron.output = self.sigmoid(neuronSum, currentNeuron.activationResponse)
            currentNeuron.output = self.sigmoid(neuronSum)
            # print("output", currentNeuron.output)
            # print("")

            if (currentNeuron.neuronType == NeuronType.OUTPUT):
                outputs.append(currentNeuron.output)

        # print("outputs:", outputs)
        return outputs
