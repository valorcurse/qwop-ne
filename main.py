from qwop import QWOP, Key
from neuralNetwork import Net
from neat import NEAT
from genes import NeuronType

import torch
from torch.autograd import Variable

import numpy as np
import cv2
import time

qwop = QWOP()
# net = Net()
# net.cuda()

qwop.grabImage()
# cv2.imshow('running track', qwop.runningTrack())
# cv2.waitKey()

print("Creating NEAT object")
neat = NEAT(12, qwop.runningTrack().size, 4)
qwop.stop()

while True:

	fitnessScores = []
	
	for phenotype in neat.phenotypes:
		qwop = QWOP()
		running = True
		gameStarted = False
		
		fitnessScore = 0

		startTime = None

		print("Network:")
		print("Hidden neurons: " + 
			str(len([neuron for neuron in phenotype.neurons if neuron.neuronType == NeuronType.HIDDEN])))

		while (running):
			qwop.grabImage()
			
			if (not qwop.isPlayable()):
				if (not gameStarted):
					gameStarted = True
					qwop.startGame()
				else:
					# fitnessScores.append()
					running = False
			else:
				# data = Variable(
				# 	torch.from_numpy(qwop.runningTrack()).type(torch.cuda.FloatTensor)
				# 	).unsqueeze(0).permute(0, 3, 1, 2)
				# outputs = net(data)
				# _, predicted = torch.max(outputs.data, 1)
				# print(predicted)
				# print(predicted[0])
				# print(Key(predicted[0]).name)
				# qwop.runningTrack()
				# print(qwop.score())
				previousFitnessScore = fitnessScore
				fitnessScore = qwop.score()

				if fitnessScore == previousFitnessScore:
					if startTime == None:
						startTime = time.time()
					else:
						print("\rTime standing still: " + str(time.time() - startTime), end='')
						if (time.time() - startTime) > 3.0:
							print("")
							print("Stopping game.")
							running = False
				else:
					startTime = None

				predicted = np.argmax(phenotype.update(qwop.runningTrack().flatten()), axis=0)

				qwop.pressKey(Key(predicted).name)

		print("Fitness score: " + str(fitnessScore))
		fitnessScores.append(fitnessScore)
		qwop.stop()

	print("Running epoch")
	neat.phenotypes = neat.epoch(fitnessScores)
	print("Generation: " + str(neat.generation))