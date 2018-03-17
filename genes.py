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

        self.currentNeuronID = 0

    def createNewLinkInnovation(self, fromNeuron, toNeuron):
        ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.LINK)

        if (ID == -1):
            newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromNeuron, toNeuron, -1,
                                        InnovationType.LINK)
            self.listOfInnovations.append(newInnovation)
            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewLink(self, fromNeuron, toNeuron, enabled, weight, recurrent=False):    
        ID = self.createNewLinkInnovation(fromNeuron, toNeuron)
        return SLinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

    def createNewNeuronInnovation(self):
        ID = innovations.checkInnovation(None, None, InnovationType.NEURON)
        
        if (ID == -1):
            values = np.array(self.listOfInnovations)
            neurons = np.where(values == InnovationType.NEURON)
            
            newNeuronID = len(neurons)
        
            newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations),
                                        None, None, newNeuronID, NeuronType.HIDDEN)

            self.listOfInnovations.append(newInnovation)

            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewNeuron(self, y, neuronType, neuronID = None):
        innovationID = self.createNewNeuronInnovation()

        if (neuronID is None):
            neuronID = self.currentNeuronID
            self.currentNeuronID += 1

        return SNeuronGene(neuronType, neuronID, y, innovationID)
    
    def checkInnovation(self, start, end, innovationType, neuronID = -1):
        matched = next((innovation for innovation in self.listOfInnovations if (
                (innovation.start == start) and
                (innovation.end == end) and
                (innovation.neuronID == neuronID) and
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

    # def getInnovation(self, innovationID):
    #     return self.listOfInnovations[innovationID] - 1



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

    def __eq__(self, other):
        return self.innovationID == other.innovationID

class SNeuronGene:
    def __init__(self, neuronType, ID, y, innovationID):
        self.neuronType = neuronType
        self.ID = ID
        self.splitY = y
        self.innovationID = innovationID
        
        self.activationResponse = None
        self.bias = 0.0


class SLink:

    def __init__(self, fromNeuron, toNeuron, weight, recurrent=False):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.recurrent = recurrent


class SNeuron:
    def __init__(self, neuronType, neuronID, bias, y, activationResponse):
        self.linksIn = []
        # self.linksOut = []

        self.sumActivation = 0.0
        self.bias = bias
        self.output = 0.0

        self.neuronType = neuronType

        self.ID = neuronID

        self.activationResponse = activationResponse

        self.posX = self.posY = 0
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
        pyplot.annotate(str("{:1.2f}".format(self.bias)), xy=(self.posX - neuron_radius/2, self.posY - neuron_radius/2))

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
        # self.currentNeuronID = outputs

        self.inputs = inputs
        self.outputs = outputs
        
        self.links = links
        self.neurons = neurons
        
        if (len(self.neurons) == 0):
            for n in range(inputs):
                newNeuron = innovations.createNewNeuron(0.0, NeuronType.INPUT, -n-1)
                # self.currentNeuronID += 1
                self.neurons.append(newNeuron)

            for n in range(outputs):
                newNeuron = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, -inputs-n-1)
                # self.currentNeuronID += 1
                self.neurons.append(newNeuron)

        self.fitness = 0

        # For printing
        self.distance = 0
        self.uniqueKeysPressed = 0
            

    def __lt__(self, other):
        return self.fitness < other.fitness

    def calculateCompatibilityDistance(self, other):
        disjointRate = 1.0
        matchedRate = 0.5

        disjointedLinks = 0.0
        weightDifferences = []

        combinedIndexes = list(set(
            [l.innovationID for l in self.links] + [l.innovationID for l in other.links]))
        combinedIndexes.sort()
        
        selfDict = {l.innovationID: l for l in self.links}
        otherDict = {l.innovationID: l for l in other.links}

        for i in combinedIndexes:
            selfLink = selfDict.get(i)
            otherLink = otherDict.get(i)

            # print("{} {}".format(selfLink.innovationID if selfLink else "null", otherLink.innovationID if otherLink else "null"))

            if (selfLink is None or otherLink is None):
                disjointedLinks += 1.0
            else:
                weightDifferences.append(math.fabs(selfLink.weight - otherLink.weight))

        longestLinks = max(1.0, max(len(other.links), len(self.links)))
        # longestLinks = 1.0 if longestLinks <= 20 else longestLinks
        weightDifference = 0.0 if len(weightDifferences) == 0 else np.mean(weightDifferences)
        # print(longestLinks)
        # print(disjointedLinks, weightDifferences)

        linkDistance = (disjointRate * disjointedLinks / longestLinks) + (matchedRate * weightDifference)

        disjointedNeurons = 0.0
        biasDifferences = []

        combinedNeurons = list(set(
            [n.innovationID for n in self.neurons] + [n.innovationID for n in other.neurons]))
        combinedNeurons.sort()
        
        selfDict = {n.innovationID: n for n in self.neurons}
        otherDict = {n.innovationID: n for n in other.neurons}

        for i in combinedNeurons:
            selfNeuron = selfDict.get(i)
            otherNeuron = otherDict.get(i)

            # print("{} {}".format(selfNeuron.innovationID if selfNeuron else "null", otherNeuron.innovationID if otherNeuron else "null"))

            if (selfNeuron is None or otherNeuron is None):
                disjointedNeurons += 1.0
            else:
                biasDifferences.append(math.fabs(selfNeuron.bias - otherNeuron.bias))

        longestNeurons = max(1.0, max(len(other.neurons), len(self.neurons)))
        # longestNeurons = 1.0 if longestNeurons <= 20 else longestNeurons
        biasDifference = 0.0 if len(biasDifferences) == 0 else np.mean(biasDifferences)

        neuronDistance = (disjointRate * disjointedNeurons / longestNeurons) + (matchedRate * biasDifference)

        return linkDistance + neuronDistance
        # return linkDistance

    def addLink(self, chanceOfLooped, triesToFindLoop, triesToAddLink):

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
        fromNeurons = [neuron for neuron in self.neurons
                       if (neuron.neuronType in [NeuronType.INPUT, NeuronType.HIDDEN])]

        toNeurons = [neuron for neuron in self.neurons
                     if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
        
        while (triesToAddLink):
            
            fromNeuron = random.choice(fromNeurons)
            toNeuron = random.choice(toNeurons)

            for l in self.links:
                if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID):
                    return

            if (fromNeuron.ID != toNeuron.ID and fromNeuron.splitY < toNeuron.splitY):
                break
            else:
                fromNeuron = toNeuron = None

            triesToAddLink -= 1

        if (fromNeuron == None or toNeuron == None):
            return

        # if (fromNeuron.splitY > toNeuron.splitY):
            # recurrent = True

        # link = innovations.createNewLink(fromNeuron, toNeuron, True, random.gauss(0.0, 1.0), recurrent)

        link = innovations.createNewLink(fromNeuron, toNeuron, True, 1.0, recurrent)
        self.links.append(link)


    def removeLink(self):
        if (len(self.links) == 0):
            return

        randomLink = random.choice(self.links)
        self.links.remove(randomLink)


    def addNeuron(self):

        if (len(self.links) < 1):
            return

        maxRand = len(self.links)

        # sizeThreshold = self.inputs + self.outputs + 1
        # if (len(self.links) < sizeThreshold):
            # maxRand = math.ceil(len(self.links) - math.sqrt(len(self.links)))

        possibleLinks = [l for l in self.links[:maxRand] if l.enabled]
            # if l.enabled and not l.recurrent]

        # print(len(possibleLinks), len(self.links))
        if (len(possibleLinks) == 0):
            return

        chosenLink = random.choice(possibleLinks)
        
        # print(chosenLink.innovationID)

        originalWeight = chosenLink.weight
        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newDepth = (fromNeuron.splitY + toNeuron.splitY) / 2

        newNeuron = innovations.createNewNeuron(newDepth, NeuronType.HIDDEN)
        # self.currentNeuronID += 1

        link1 = innovations.createNewLink(fromNeuron, newNeuron, True, 1.0)
        self.links.append(link1)
        
        link2 = innovations.createNewLink(newNeuron, toNeuron, True, originalWeight)
        self.links.append(link2)

        self.links.remove(chosenLink)

        # print(self.ID, "-- Adding new neuron:", newNeuron.ID)
        # print(self.ID, "-- Links:", [(l.innovationID, l.fromNeuron.ID, l.toNeuron.ID) for l in self.links])

        self.neurons.append(newNeuron)
        self.neurons.sort(key=lambda x: x.splitY, reverse=False)

        # print([n.ID for n in self.neurons])
        # print(len(self.links), len(self.neurons))

    def removeNeuron(self):
        possibleNeurons = [n for n in self.neurons if n.neuronType == NeuronType.HIDDEN]
        if (len(possibleNeurons) == 0):
            return

        randomNeuron = random.choice(possibleNeurons)

        # allLinks = randomNeuron.linksIn + randomNeuron.linksOut
        connectedLinks = [l for l in self.links if l.toNeuron.ID == randomNeuron.ID or l.fromNeuron.ID == randomNeuron.ID]
        # print(connectedLinks)
        for link in connectedLinks:
            self.links.remove(link)

        self.neurons.remove(randomNeuron)

    def mutateWeights(self, mutationRate, replacementProbability, maxWeightPerturbation):
            for link in self.links:
                if (random.random() < mutationRate):
                    # link.weight += random.uniform(-1.0, 1.0) * maxWeightPerturbation
                    link.weight += random.gauss(0.0, maxWeightPerturbation)
                    link.weight = min(30.0, max(-30.0, link.weight))

                elif (random.random() < replacementProbability):
                    link.weight = random.gauss(0.0, 1.0)


    def mutateBias(self, mutationRate, replacementProbability, maxWeightPerturbation):
        # biasInput = [n for n in self.neurons if n.neuronType == NeuronType.HIDDEN][0]

        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() < mutationRate):
                n.bias += random.gauss(0.0, maxWeightPerturbation)
                n.bias = min(30.0, max(-30.0, n.bias))

            elif (random.random() < replacementProbability):
                n.bias = random.gauss(0.0, maxWeightPerturbation)

    def createPhenotype(self):
        phenotypeNeurons = []

        # print("--", self.ID)
        # print(len(self.neurons))
        # print("--------- Genome", self.ID)
        # print([n.ID for n in self.neurons])        
        # print([n.innovationID for n in self.neurons])        
        for neuron in self.neurons:
            newNeuron = SNeuron(neuron.neuronType,
                        neuron.ID,
                        # neuron.innovationID,
                        neuron.bias,
                        neuron.splitY,
                        neuron.activationResponse)

            phenotypeNeurons.append(newNeuron)



        for link in self.links:
            if (link.enabled):
                fromNeuron = next((neuron
                                   for neuron in phenotypeNeurons if (neuron.ID == link.fromNeuron.ID)), None)
                toNeuron = next((neuron
                                 for neuron in phenotypeNeurons if (neuron.ID == link.toNeuron.ID)), None)

                # print([n.ID for n in self.neurons])
                # print([neuron.ID for neuron in phenotypeNeurons if (neuron.ID == link.toNeuron.ID)])
                # print(link.fromNeuron.ID, link.toNeuron.ID)
                # print(fromNeuron.ID, toNeuron.ID)

                tmpLink = SLink(fromNeuron,
                                toNeuron,
                                link.weight,
                                link.recurrent)

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
        pyplot.draw()
        pyplot.pause(0.5)
        pyplot.show()

    # def sigmoid(self, x):
        # return 1.0 / (1.0 + math.exp(-x))

    def sigmoid(self, z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return 1.0 / (1.0 + math.exp(-z))

    # Actually tanh
    # def sigmoid(self, x):
        # return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))

    def update(self, inputs):
        # outputs = []

        table = PrettyTable(["Type", "ID", "output"])
        
        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value
            # table.add_row([neuron.neuronType, neuron.ID, neuron.output])

        for currentNeuron in self.neurons[len(inputNeurons):]:
            neuronSum = []
            for link in currentNeuron.linksIn:
                neuronSum.append(link.fromNeuron.output * link.weight)

            currentNeuron.output = self.sigmoid(currentNeuron.bias + sum(neuronSum))
            table.add_row([currentNeuron.neuronType, currentNeuron.ID, currentNeuron.output])

            # if (currentNeuron.neuronType == NeuronType.OUTPUT):
                # outputs.append(currentNeuron.output)

        # print(table)
        return [n.output for n in self.neurons if n.neuronType == NeuronType.OUTPUT]

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