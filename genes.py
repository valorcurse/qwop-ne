import random
from random import randint

import math
from math import cos, sin, atan, ceil, floor
from sklearn.preprocessing import normalize
from enum import Enum
import itertools

import numpy as np
from matplotlib import pyplot

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
                                    NeuronType.LINK)
        self.listOfInnovations.append(newInnovation)

        return len(self.listOfInnovations) - 1;

    def createNewLink(self, fromNeuron, toNeuron, enabled, weight, recurrent=False):
        ID = innovations.checkInnovation(fromNeuron, toNeuron, NeuronType.LINK)
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

    def calculateCompatibilityDistance(self, otherGenome):
        numExcess = 0.0
        numMatched = 1.0
        numDisjointed = 0.0

        weightDifference = 0.0

        combinedLinks = itertools.zip_longest(self.links, otherGenome.links)

        for selfLink, otherLink in combinedLinks:

            if (not selfLink):
                numExcess += 1
                continue

            if (not otherLink):
                numExcess += 1
                continue

            selfID = selfLink.innovationID
            otherID = otherLink.innovationID

            if (selfID == otherID):
                numMatched += 1
                weightDifference += math.fabs(selfLink.weight - otherLink.weight)

            if (selfID < otherID):
                numDisjointed += 1

            if (selfID > otherID):
                numDisjointed += 1

        longest = max(len(otherGenome.links), len(self.links))
        longest = 1.0 if longest <= 20.0 else longest

        disjoint = 1.0
        excess = 1.0
        matched = 0.4

        return ((excess * numExcess / longest) +
                (disjoint * numDisjointed / longest) +
                (matched * weightDifference / numMatched))

    def addLink(self, mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink):

        if (random.random() > mutationRate):
            return

        fromNeuron = None
        toNeuron = None
        recurrent = False

        # Add recurrent link
        if (random.random() < chanceOfLooped and len(self.neurons) > (self.inputs + self.outputs)):
            possibleNeurons = [n for n in self.neurons[self.inputs + 1:len(self.neurons) - 1]
                if not n.recurrent or n.neuronType != NeuronType.BIAS or n.neuronType != NeuronType.INPUT]

            if (len(possibleNeurons) == 0):
                return

            loopNeuron = random.choice(possibleNeurons)
            fromNeuron = toNeuron = loopNeuron
            recurrent = loopNeuron.recurrent = True

        else:
            loopFound = False
            while (triesToAddLink and not loopFound):
                fromNeurons = [neuron for neuron in self.neurons
                               if (neuron.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.HIDDEN])]
                fromNeuron = random.choice(fromNeurons)

                toNeurons = [neuron for neuron in self.neurons
                             if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
                toNeuron = random.choice(toNeurons)

                linkIsDuplicate = next(
                    (l for l in self.links
                     if (l.fromNeuron == fromNeuron) and
                     (l.toNeuron == toNeuron)),
                    None)

                if (not linkIsDuplicate or fromNeuron.ID == toNeuron.ID):
                    loopFound = True
                else:
                    fromNeuron = toNeuron = None

                triesToAddLink -= 1

        if (fromNeuron == None or toNeuron == None):
            return

        if (fromNeuron.splitY > toNeuron.splitY):
            recurrent = True

        randomClamped = random.random() - random.random()
        link = innovations.createNewLink(fromNeuron, toNeuron, True, randomClamped, recurrent)
        self.links.append(link)


    def addNeuron(self, chanceToAddNeuron, triesToFindOldLink):

        if (len(self.links) < 2):
            return

        if (random.random() > chanceToAddNeuron):
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
        
        link2 = innovations.createNewLink(fromNeuron, newNeuron, True, originalWeight)
        self.links.append(link2)

        self.neurons.append(newNeuron)
        self.neurons.sort(key=lambda x: x.splitY, reverse=False)

    def mutateWeights(self, mutationRate, replacementProbability, maxWeightPerturbation):
        if (random.random() < mutationRate):
            for link in self.links:
                if (random.random() < replacementProbability):
                    link.weight = random.random()
                    return
                else:
                    link.weight += random.uniform(-1, 1) * maxWeightPerturbation

    def mutateBias(self, mutationRate, maxWeightPerturbation):
        if (random.random() < mutationRate):
            biasInput = [n for n in self.neurons if n.neuronType == NeuronType.BIAS][0]
            biasInput.biasValue += random.uniform(-1, 1) * maxWeightPerturbation

    def createPhenotype(self, depth):
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

        return CNeuralNet(phenotypeNeurons, depth, self.ID, self)


class CNeuralNet:

    def __init__(self, neurons, depth, ID, genome):
        self.genome = genome

        self.neurons = neurons
        self.depth = depth
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
        pyplot.pause(1)
        # pyplot.show()

    def draw(self):
        # print("Phenotype", self.ID)
        # print("Neurons:")
        # for n in self.neurons:
        #     print("ID:", n.ID, "\tType:", n.neuronType, 
        #         "  \tx:", n.splitY, "\ty:", n.splitX)

        #     for l in n.linksIn:
        #             print("\t", l.fromNeuron.ID, "->", l.toNeuron.ID, 
        #                 "\tWeight:", l.weight, "\tRecurrent:", l.recurrent)

        # print("")

        # print("Genome", self.ID)
        # print("Neurons:")
        # for n in self.genome.neurons:
        #     print("ID:", n.ID, "InnovationID:", n.innovationID, "\tType:", n.neuronType, 
        #         "  \tx:", n.splitY, "\ty:", n.splitX)
        # print("Links:")
        # for l in self.genome.links:
        #     if l.enabled:
        #         print("ID:", l.innovationID, "\t", l.fromNeuron.ID, "->", l.toNeuron.ID, 
        #             "\tWeight:", l.weight, "    \tEnabled:", l.enabled, "\tRecurrent:", l.recurrent)

        pyplot.clf()
        for layer in self.layers:
            layer.draw()

        pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
        pyplot.axis('scaled')
        pyplot.gca().relim()
        pyplot.gca().autoscale_view()
        pyplot.draw()
        pyplot.pause(0.5)
        # pyplot.show()

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def update(self, inputs):
        outputs = []

        inputs = inputs / 255

        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        # Set bias
        # self.neurons[len(inputNeurons)].output = 1.0

        for currentNeuron in self.neurons[len(inputNeurons)+1:]:
            neuronSum = 0.0
            for link in currentNeuron.linksIn:
                weight = link.weight

                neuronOutput = link.fromNeuron.output
                neuronSum += weight * neuronOutput
                
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