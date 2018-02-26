import random
from random import randint

import math
from math import cos, sin, atan, ceil, floor
from sklearn.preprocessing import normalize
from enum import Enum
import itertools

import numpy as np
from matplotlib import pyplot

from prettytable import PrettyTable

image_width = 34
image_height = 31
horizontal_distance_between_layers = 5
vertical_distance_between_neurons = 5
neuron_radius = 1
number_of_neurons_in_widest_layer = 4

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

    def __eq__(self, other):
        return self.innovationType == other

class Innovations:
    def __init__(self):
        self.listOfInnovations = []

        self.currentNeuronID = -1

    def createNewLinkInnovation(self, fromNeuron, toNeuron):
        newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromNeuron, toNeuron, -1,
                                    InnovationType.LINK)
        self.listOfInnovations.append(newInnovation)

        return len(self.listOfInnovations) - 1;

    def createNewLink(self, fromNeuron, toNeuron, enabled, weight, recurrent=False):
        ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.LINK)
        if (ID == -1):
            ID = self.createNewLinkInnovation(fromNeuron, toNeuron)
        
        return SLinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

    def createNewNeuronInnovation(self, fromNeuron, toNeuron):
        # neurons = [neuron for neuron in self.listOfInnovations if neuron.innovationType == InnovationType.NEURON]
        values = np.array(self.listOfInnovations)
        neurons = np.where(values == InnovationType.NEURON)
        newNeuronID = len(neurons)
        newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations),
                                    fromNeuron, toNeuron, newNeuronID, NeuronType.HIDDEN)
        self.listOfInnovations.append(newInnovation)

        return len(self.listOfInnovations) - 1;

    def createNewNeuron(self, fromNeuron, toNeuron, x, y, neuronType):
        # ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.NEURON)
        # if (ID == -1) and (innovations.getInnovation(ID) != None):
        ID = self.createNewNeuronInnovation(fromNeuron, toNeuron)
        self.currentNeuronID += 1

        return SNeuronGene(neuronType, self.currentNeuronID, x, y, ID)
    
    def checkInnovation(self, start, end, innovationType):
        matched = next((innovation for innovation in self.listOfInnovations if (
                (innovation.start == start) and
                (innovation.end == end) and
                (innovation.innovationType == innovationType))), None)

        # table = PrettyTable(["start", "end", "type", "matched"])
        # table.add_row([start.ID, end.ID, innovationType, None])
        # for innovation in self.listOfInnovations:
        #     if (innovation.innovationType == InnovationType.NEURON):
        #         continue

        #     isMatched = (innovation.start == start) and (innovation.end == end) and (innovation.innovationType == innovationType)
        #     table.add_row([innovation.start.ID if innovation.start else "none", 
        #         innovation.end.ID if innovation.end else "none", 
        #         innovation.innovationType, 
        #         matched])

        # print(table)

        # print("matched:", matched)
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
    # def __init__(self, neuronType, ID, x, y, innovationID, recurrent=False):
    def __init__(self, neuronType, ID, x, y, innovationID):
        self.ID = ID
        self.neuronType = neuronType
        self.recurrent = False
        self.activationResponse = None
        self.splitX = x
        self.splitY = y

        # In case it's a bias neuron
        self.biasValue = 1.0

        self.innovationID = innovationID

        self.numInputs = 0
        self.numOutputs = 0

        self.neurons = []
        self.links = []


class SLink:

    def __init__(self, fromNeuron, toNeuron, weight, recurrent=False):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

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

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = 0.0
        if (neuron1[0] != neuron2[0] and neuron1[1] != neuron2[1]):
            angle = atan((neuron2[0] - neuron1[0]) / float(neuron2[1] - neuron1[1]))
            # return

        x_adjustment = 0
        y_adjustment = 0
        line = pyplot.Line2D(
            (neuron1[0] - x_adjustment, neuron2[0] + x_adjustment), 
            (neuron1[1] - y_adjustment, neuron2[1] + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self):
        # print("Drawing neuron", self.posX, self.posY)
        circle = pyplot.Circle(
            (self.posX, self.posY), 
            radius=neuron_radius, fill=(self.neuronType == NeuronType.BIAS))
        pyplot.gca().add_patch(circle)
        pyplot.annotate(str(self.ID), xy=(self.posX - neuron_radius/4, self.posY))
        pyplot.annotate(str("{:1.2f}".format(self.output)), xy=(self.posX - neuron_radius/2, self.posY - neuron_radius/2))

        for l in self.linksIn:
            fromNeuron = l.fromNeuron
            self.__line_between_two_neurons(
                (fromNeuron.posX, fromNeuron.posY), 
                (self.posX, self.posY))

            pyplot.annotate(str("{:1.2f}".format(l.weight)), xy=(self.posX - (self.posX - fromNeuron.posX)/2, self.posY - (self.posY - fromNeuron.posY)/2))


            pyplot.pause(0.005)

class CGenome:

    def __init__(self, ID, neurons, links, inputs, outputs):
        self.ID = ID
        self.neurons = neurons
        self.links = links
        self.inputs = inputs
        self.outputs = outputs
        
        self.fitness = 0

        # For printing
        self.distance = 0
        self.uniqueKeysPressed = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def calculateCompatibilityDistance(self, other):
        numDisjointed = 0
        weightDifferences = []

        combinedIndexes = list(set(
            [l.innovationID for l in self.links] + [l.innovationID for l in other.links]))
        combinedIndexes.sort()
        
        selfDict = {l.innovationID: l for l in self.links}
        otherDict = {l.innovationID: l for l in other.links}

        # print("-------------------------------------------------")
        babyLinks = []
        for i in combinedIndexes:
            selfLink = selfDict.get(i)
            otherLink = otherDict.get(i)

            # print("{} {}".format(selfLink.innovationID if selfLink else "null", otherLink.innovationID if otherLink else "null"))

            if (selfLink is None or otherLink is None):
                numDisjointed += 1
            else:
                selfID = selfLink.innovationID
                otherID = otherLink.innovationID

                if (selfID == otherID):
                    weightDifferences.append(math.fabs(selfLink.weight - otherLink.weight))

        longest = max(1.0, max(len(other.links), len(self.links)))
        # print("longest:", longest)
        # longest = 1.0 if longest <= 20.0 else longest

        weightDifference = 0.0 if len(weightDifferences) == 0 else np.mean(weightDifferences)

        # print(numMatched, numDisjointed, numExcess, weightDifferences)

        disjoint = 1.0
        matched = 0.5

        disjointDelta = (disjoint * numDisjointed / longest)
        matchedDelta = (matched * weightDifference)

        # print(excessDelta, disjointDelta, matchedDelta, excessDelta + disjointDelta + matchedDelta)

        return disjointDelta + matchedDelta

    def addLink(self, mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink):

        if (random.random() > mutationRate):
            return

        fromNeuron = None
        toNeuron = None
        recurrent = False

        # Add recurrent link
        # if (random.random() < chanceOfLooped and len(self.neurons) > (self.inputs + self.outputs)):
        #     possibleNeurons = [n for n in self.neurons[self.inputs + 1:len(self.neurons) - 1]
        #         if not n.recurrent or n.neuronType != NeuronType.BIAS or n.neuronType != NeuronType.INPUT]

        #     if (len(possibleNeurons) == 0):
        #         return

        #     # loopNeuron = random.choice(possibleNeurons)
        #     # fromNeuron = toNeuron = loopNeuron
        #     # recurrent = loopNeuron.recurrent = True

        # else:
        keepLoopRunning = True
        while (triesToAddLink):
            fromNeurons = [neuron for neuron in self.neurons
                           if (neuron.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.HIDDEN])]
            # fromNeurons = [neuron for neuron in self.neurons
                           # if (neuron.neuronType in [NeuronType.BIAS])]
            fromNeuron = random.choice(fromNeurons)

            toNeurons = [neuron for neuron in self.neurons
                         if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
            toNeuron = random.choice(toNeurons)

            linkIsDuplicate = False

            # print("ID:", self.ID)
            for l in self.links:
                # print([l.fromNeuron.ID, l.toNeuron.ID], "==", [fromNeuron.ID, toNeuron.ID])
                if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID):
                    linkIsDuplicate = True
                    # print("Found duplicate")
                    break

            # linkIsDuplicate = next(
            #     (l for l in self.links
            #      if (l.fromNeuron == fromNeuron) and
            #      (l.toNeuron == toNeuron)),
            #     None)

            # print("Duplicate:", fromNeuron.ID, toNeuron.ID, linkIsDuplicate)

            if (not linkIsDuplicate and fromNeuron.ID != toNeuron.ID and fromNeuron.splitY < toNeuron.splitY):
                break
            else:
                fromNeuron = toNeuron = None

            triesToAddLink -= 1

        if (fromNeuron == None or toNeuron == None):
            return

        if (fromNeuron.splitY > toNeuron.splitY):
            recurrent = True

        link = innovations.createNewLink(fromNeuron, toNeuron, True, random.gauss(0.0, 1.0), recurrent)
        # print("Adding link:", [fromNeuron.ID, toNeuron.ID])
        self.links.append(link)


    def addNeuron(self, chanceToAddNeuron):

        if (len(self.links) < 1):
            return

        randomChance = random.random()
        if (randomChance > chanceToAddNeuron):
            # print(randomChance)
            return

        maxRand = len(self.links)

        sizeThreshold = self.inputs + self.outputs + 5
        if (len(self.links) < sizeThreshold):
            maxRand = math.floor(len(self.links) - math.sqrt(len(self.links)))

        possibleLinks = [l for l in self.links[:maxRand]
            if l.enabled and not l.recurrent and l.fromNeuron.neuronType != NeuronType.BIAS]

        if (len(possibleLinks) == 0):
            return

        chosenLink = random.choice(possibleLinks)
        chosenLink.enabled = False

        originalWeight = chosenLink.weight

        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newDepth = (fromNeuron.splitY + toNeuron.splitY) / 2
        newWidth = (fromNeuron.splitX + toNeuron.splitX) / 2

        newNeuron = innovations.createNewNeuron(fromNeuron, toNeuron, newWidth, newDepth, NeuronType.HIDDEN)

        link1 = innovations.createNewLink(fromNeuron, newNeuron, True, 1.0)
        self.links.append(link1)
        
        link2 = innovations.createNewLink(newNeuron, toNeuron, True, originalWeight)
        self.links.append(link2)

        self.neurons.append(newNeuron)
        self.neurons.sort(key=lambda x: x.splitY, reverse=False)

    def mutateWeights(self, mutationRate, replacementProbability, maxWeightPerturbation):
            for link in self.links:
                if (random.random() < replacementProbability):
                    link.weight = random.gauss(0.0, 1.0)
                elif (random.random() < mutationRate):
                    # link.weight += random.uniform(-1.0, 1.0) * maxWeightPerturbation
                    link.weight += random.gauss(0.0, maxWeightPerturbation)
                    link.weight = min(1.0, max(-1.0, link.weight))


    def mutateBias(self, mutationRate, maxWeightPerturbation):
        if (random.random() < mutationRate):
            biasInput = [n for n in self.neurons if n.neuronType == NeuronType.BIAS][0]

            # biasInput.biasValue += random.uniform(-1.0, 1.0) * maxWeightPerturbation
            biasInput.biasValue += random.gauss(0.0, maxWeightPerturbation)
            biasInput.biasValue = min(1.0, max(-1.0, biasInput.biasValue))

    def createPhenotype(self):
        phenotypeNeurons = []

        # print("Genome", self.ID)
        # print("Neurons:")
        # for n in self.neurons:
        #     print("ID:", n.ID, "InnovationID:", n.innovationID, "\tType:", n.neuronType, 
        #         "  \tx:", n.splitY, "\ty:", n.splitX)
        # print("Links:")
        # for l in self.links:
        #     if l.enabled:
        #         print("ID:", l.innovationID, "\t", l.fromNeuron.ID, "->", l.toNeuron.ID, 
        #             "\tWeight:", l.weight, "    \tEnabled:", l.enabled, "\tRecurrent:", l.recurrent)

        # print("")
        for neuron in self.neurons:
            newNeuron = SNeuron(neuron.neuronType,
                        neuron.ID,
                        neuron.splitY,
                        neuron.splitX,
                        neuron.activationResponse)

            if (neuron.neuronType == NeuronType.BIAS):
                newNeuron.output = neuron.biasValue

            phenotypeNeurons.append(newNeuron)



        for link in self.links:
            if (link.enabled):
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

        return CNeuralNet(phenotypeNeurons, self.ID, self)


class CNeuralNet:

    def __init__(self, neurons, ID, genome):
        self.genome = genome

        self.neurons = neurons
        self.ID = ID

        self.toDraw = False
        self.layers = []
        uniqueDepths = sorted(set([n.splitY for n in self.neurons]))
        # print("Depths:", uniqueDepths)
        for d in uniqueDepths:
            # if d == 0:
            #     continue

            # neuronsToDraw = [n for n in self.neurons 
            #     if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT] and n.splitY == d]
            neuronsToDraw = [n for n in self.neurons if n.splitY == d]
            self.layers.append(Layer(self, neuronsToDraw))



    def draw(self, image):
        pyplot.clf()
        pyplot.imshow(image, cmap='gray')
        pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
        for layer in self.layers:
            layer.draw()

        # pyplot.draw()
        pyplot.axis('scaled')
        # pyplot.pause(1)
        pyplot.show()

    def draw(self):
        pyplot.clf()
        for layer in self.layers:
            layer.draw()

        pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
        pyplot.axis('scaled')
        pyplot.gca().relim()
        pyplot.gca().autoscale_view()
        # pyplot.draw()
        # pyplot.pause(0.5)
        pyplot.show()

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def update(self, inputs):
        outputs = []

        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        # Set bias
        # self.neurons[len(inputNeurons)].output = 1.0
        # print("Inputs:", inputs)
        for currentNeuron in self.neurons[len(inputNeurons)+1:]:

            # print("Updating network:", currentNeuron.neuronType)
            neuronSum = 0.0
            for link in currentNeuron.linksIn:
                weight = link.weight

                neuronOutput = link.fromNeuron.output
                # neuronSum += round(weight * neuronOutput, 5)
                neuronSum += weight * neuronOutput

                # print(weight, "*", neuronOutput, "=", neuronSum)
                
            currentNeuron.output = self.sigmoid(neuronSum)

            if (currentNeuron.neuronType == NeuronType.OUTPUT):
                outputs.append(currentNeuron.output)

        return outputs

class Layer():
    def __init__(self, network, neuronsToDraw):
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        
        self.neurons = neuronsToDraw
        self.__intialise_neurons()

    def __intialise_neurons(self):
        startY = self.__calculate_top_margin_so_layer_is_centered(len(self.neurons))
        for neuron in self.neurons:
            neuron.posX = self.x
            neuron.posY = startY
            startY += vertical_distance_between_neurons

    def __calculate_top_margin_so_layer_is_centered(self, number_of_neurons):
        return vertical_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + horizontal_distance_between_layers
        else:
            return horizontal_distance_between_layers + neuron_radius

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()