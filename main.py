from qwop import QWOP, Key
from neuralNetwork import Net
from neat import NEAT

import torch
from torch.autograd import Variable

qwop = QWOP()
net = Net()
net.cuda()

qwop.grabImage()

neat = NEAT(12, qwop.runningTrack().size, 4)

while True:

	fitnessScores = []
	
	for phenotype in neat.phenotypes:
		running = True
		gameStarted = False
		
		while (running):
			qwop.grabImage()
			
			fitnessScore = 0

			if (not qwop.isPlayable()):
				if (not gameStarted):
					gameStarted = True
					qwop.startGame()
				else:
					fitnessScores.append()
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
				print(qwop.runningTrack().shape)
				fitnessScore = qwop.score()

				predicted = math.max(phenotype.update(qwop.runningTrack()))

				qwop.pressKey(Key(predicted).name)

	neat.phenotypes = neat.epoch(fitnessScores)