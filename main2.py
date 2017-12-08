from qwop import QWOP, Key
from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

from genes import innovations
global innovations

import torch
from torch.autograd import Variable

import numpy as np
import cv2
import time
from random import randint

print("Creating NEAT object")
neat = NEAT(50, 5, 4)

while True:
	print("----------------------------------------")
	
	fitnessScores = []
	
	for phenotype in neat.phenotypes:
		running = True
		gameStarted = False
		
		startTime = None

		# print("Network:")
		# print("Hidden neurons: " + 
			# str(len([neuron for neuron in phenotype.neurons if neuron.neuronType == NeuronType.HIDDEN])))

		phenotype.update([randint(1, 10)]*5)
		fitnessScores.append(randint(1, 100))

	print("Running epoch")
	print("Fitness scores: ", fitnessScores)
	neat.phenotypes = neat.epoch(fitnessScores)
	print("Generation: " + str(neat.generation))
	print("Number of innovations: " + str(len(innovations.listOfInnovations)))
	print("Number of phenotypes: " + str(len(neat.phenotypes)))


	time.sleep(1)