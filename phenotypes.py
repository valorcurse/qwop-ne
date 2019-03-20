from enum import Enum

import math
from math import cos, sin, atan, ceil, floor

import numpy as np
from matplotlib import pyplot
import matplotlib.patches as patches

image_width = 34
image_height = 31
horizontal_distance_between_layers = 5
vertical_distance_between_neurons = 5
neuron_radius = 1
number_of_neurons_in_widest_layer = 4

class NeuronType(Enum):
    INPUT = 0
    HIDDEN = 1
    BIAS = 2
    OUTPUT = 3
    LINK = 4

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
            if (l.recurrent):
                patchArrow = patches.FancyArrowPatch(
                    (self.posX + 0.3, self.posY),
                    (self.posX - 0.4, self.posY),
                    connectionstyle='arc3, rad=-4.0',    # Default
                    mutation_scale=5,
                    color="red"
                )
                pyplot.gca().add_patch(patchArrow)
            else:
                self.__line_between_two_neurons(
                    (fromNeuron.posX, fromNeuron.posY), 
                    (self.posX, self.posY))

            pyplot.annotate(str("{:1.2f}".format(l.weight)), xy=(self.posX - (self.posX - fromNeuron.posX)/2, self.posY - (self.posY - fromNeuron.posY)/2))


            pyplot.pause(0.005)


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

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    # def sigmoid(self, z):
    #     z = max(-60.0, min(60.0, 5.0 * z))
    #     return 1.0 / (1.0 + math.exp(-z))

    # Actually tanh
    def sigmoid(self, x):
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))

    def calcOutput(self, neuron):
        linksIn = neuron.linksIn
        if (len(linksIn) > 0):
            return self.sigmoid(neuron.bias + np.sum([self.calcOutput(linkIn.fromNeuron) * linkIn.weight for linkIn in linksIn]))
        else:
            return neuron.output

    def updateRecursively(self, inputs):
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        outputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT]

        return [self.calcOutput(outputNeuron) for outputNeuron in outputNeurons]

    def update(self, inputs):
        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        for currentNeuron in self.neurons[len(inputNeurons):]:
            linksIn = currentNeuron.linksIn
            output = np.sum([link.fromNeuron.output * link.weight for link in linksIn])
            currentNeuron.output = self.sigmoid(currentNeuron.bias + output)

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