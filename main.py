from qwop import QWOP, Key
from neuralNetwork import Net

import torch
from torch.autograd import Variable

qwop = QWOP()
net = Net()
net.cuda()

while True:
	qwop.grabImage()

	if (not qwop.isPlayable()):
		qwop.startGame()
	else:
		data = Variable(
			torch.from_numpy(qwop.runningTrack()).type(torch.cuda.FloatTensor)
			).unsqueeze(0).permute(0, 3, 1, 2)
		outputs = net(data)
		_, predicted = torch.max(outputs.data, 1)
		# print(predicted)
		# print(predicted[0])
		# print(Key(predicted[0]).name)
		# qwop.runningTrack()
		# print(qwop.score())
		qwop.pressKey(Key(predicted[0]).name)
