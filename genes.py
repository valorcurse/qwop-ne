import random
from random import randint

import math
from enum import Enum

class Innovations:
	listOfInnovations = []

	def createNewLinkInnovation(self, start, end):
		newInnovation = Innovation(InnovationType.LINK, len(listOfInnovations), start, end, -1, NeuronType.LINK)
		listOfInnovations.append(newInnovation)

		return len(listOfInnovations) - 1;

	def createNewNeuronInnovation(self, start, end, width, depth):
		neurons = [neuron for neuron in listOfInnovations if neuron.innovationType == InnovationType.NEURON]
		newNeuronID = len(neurons)
		newInnovation = Innovation(NeuronType.NEURON, len(listOfInnovations), start, end, newNeuronID. NeuronType.HIDDEN)
		listOfInnovations.append()

	def checkInnovation(self, start, end, innovationType):
		matched = [index for index, innovation in listOfInnovations if ((innovation.start == start) and (innovation.end == end) and (innovation.innovationType == innovationType))]

		return matched

	def getInnovation(self, innovationID):
		return listOfInnovations[innovationID]

innovations = Innovations()

class SLinkGene:

	def __init__(self, inNeuron, outNeuron, enabled, innovationID, weight, recurrent=False):
		self.fromNeuron = inNeuron
		self.toNeuron = outNeuron

		self.weight = weight

		self.enabled = enabled

		self.recurrent = recurrent

		self.innovationID = innovationID

	def __lt__(self, other):
		return self.innovationID < other.innovationID

class NeuronType(Enum):
	INPUT = 0
	HIDDEN = 1
	BIAS = 2
	OUTPUT = 3
	LINK = 4

class SNeuronGene:
	def __init__(self, neuronType, ID, x, y, recurrent = False):
		self.ID = ID
		self.neuronType = neuronType
		self.recurrent = recurrent
		self.activationResponse = None
		self.splitX = x
		self.splitY = y

		self.numInputs = 0
		self.numOutputs = 0

		self.neurons = []
		self.links = []

class InnovationType(Enum):
	NEURON = 0
	LINK = 1

class SInnovation:
	def __init__(self, innovationType, innovationID, start, end, neuronID, neuronType):
		self.innovationType = innovationType
		self.innovationID = innovationID
		self.start = start
		self.end = end
		self.neuronID = neuronID
		self.neuronType = neuronType

class SLink:
	
	def __init__():
		self.neuronIn = None
		self.neuronOut = None

		self.weight = 0

		self.recurrent = False

class SNeuron:
	def __init__(self, neuronType, neuronID, y, x, activationResponse):
		self.linksIn = []
		self.linksOut = []

		self.sumActivation = 0
		self.output = 0

		self.neuronType = neuronType

		self.neuronID = neuronID

		self.activationResponse = activationResponse

		self.posX = self.posY = 0
		self.splitX = x
		self.splitY = y


class CGenome:

	def __init__(self, ID, neurons, genes, inputs, outputs):
		self.genomeID = ID
		self.neurons = neurons
		self.links = genes

	def addLink(self, mutationRate, chanceOfLooped, innovation, 
		triesToFindLoop, triesToAddLink):

		if (random.random() > mutationRate):
			return

		neuron1 = -1
		neuron2 = -1
		recurrent = False

		if (random.random() < chanceOfLooped):
			loopFound = False
			while(triesToFindLoop and not loopFound):
				neuronPosition = randint(numInputs + 1, len(neurons) - 1)

				loopNeuron = neurons[neuronPosition]
				if (not loopNeuron.recurrent or 
					loopNeuron.neuronType != NeuronType.BIAS or
					loopNeuron.neuronType != NeuronType.INPUT):

						neuron1 = neuron2 = loopNeuron.ID

						recurrent = loopNeuron.recurrent = True

						loopFound = True
		else:
			loopFound = False
			while(triesToAddLink and not loopFound):
				neuron1 = neurons[randin(0, len(neurons - 1))]
				neuron2 = neurons[randin(1, len(neurons - 1))]

				if (neuron2.ID is 2):
					continue

				linkIsDuplicate = next(
					(l for l in links 
						if (l.fromNeuron != neuron1) and (l.toNeuron != neuron2) or (l.fromNeuron != neuron2) and (l.toNeuron != neuron1)), 
					None)

				if (not linkIsDuplicate
					or neuron1.ID is neuron2.ID):

						loopFound = True

				else:
					neuron1.ID = neuron2.ID = -1


		if (neuron1.ID < 0 or neuron2.ID < 0):
			return

		ID = innovations.checkInnovation(neuron1, neuron2, InnovationType.LINK)
		
		if (neuron1.splitY > neuron2.splitY):
			recurrent = True

		if (ID < 0):
			ID = innovation.createNewLinkInnovation(neuron1, neuron2)

			randomClamped = random.random() - random.random()
			newGene = SLinkGene(neuron1, neuron2, true, id, randomClamped, recurrent)
			links.append(newGene)

		else:
			newGene = SLinkGene(neuron1, neuron2, true, id, randomClamped, recurrent)
			links.append(newGene)

	def addNeuron(self, mutationRate, innovations, triesToFindOldLink):

		if (random.random() > mutationRate):
			return

		done = False
		chosenLink = 0

		sizeThreshold = self.numInputs + self.numOutputs + 5

		if (len(links) < sizeThreshold):

			loopFound = False
			while (triesToFindOldLink and not loopFound):
				# Genes might not be the same as links
				chosenLink = links[randint(0, len(links) - 1 - math.sqrt(len(links)))]

				fromNeuron = chosenLink.fromNeuron

				if (chosenLink.enabled and
					not chosenLink.recurrent and
					neurons.index(fromNeuron).neuronType != NeuronType.BIAS):

						done = loopFound = True

			if (not done):
				return

		else:

			while(not done):
				chosenLink = links[randint(0, len(links) - 1)]
				fromNeuron = chosenLink.fromNeuron

				if (chosenLink.enabled and
					not chosenLink.recurrent and
					neurons.index(fromNeuron).neuronType is not NeuronType.BIAS):
						done = True

				chosenLink.enabled = True

				originalWeight = chosenLink.weight
				
				fromNeuron = chosenLink.fromNeuron
				toNeuron = chosenLink.toNeuron

				newDepth = (neurons.index().splitY +
					neurons.index(chosenLink.toNeuron).splitY) / 2

				newWidth = (neurons.index(chosenLink.fromNeuron).splitX +
					neurons.index(chosenLink.toNeuron).splitX) / 2

				ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.NEURON)

				if (ID >= 0):
					neuronID = innovations.getInnovation(ID).neuronID

					if (neurons.index(neuronID) >= 0):
						ID = -1


				if (ID < 0):
					newNeuronID = innovations.createNewNeuronInnovation(fromNeuron, toNeuron, newWidth, newDepth)

					neurons.append(SNeuronGene(NeuronType.HIDDEN,
						newNeuronID,
						newDepth,
						newWidth))

					idLink1 = innovations.createNewLinkInnovation(fromNeuron, newNeuronID)

					link1 = SLinkGene(
						fromNeuron,
						newNeuronID,
						true,
						idLink1,
						1.0)


					idLink2 = innovations.createNewLinkInnovation(newNeuronID, toNeuron)
					link2 = SLinkGene(
						newNeuronID,
						toNeuron,
						true,
						idLink2,
						originalWeight)

					self.links.push(link2)

				else:

					newNeuronID = innovations.getInnovation(ID).neuronID

					idLink1 = innovations.checkInnovation(fromNeuron, newNeuronID, NeuronType.LINK)
					idLink2 = innovations.checkInnovation(newNeuronID, toNeuron, NeuronType.LINK)


					if (idLink1 < 0 or idLink2 < 0):
						return

					link1 = SLinkGene(fromNeuron, newNeuronID, true, idLink1, 1.0)
					link2 = SLinkGene(newNeuronID, toNeuron, true, idLink2, originalWeight)

					self.links.append(link1)
					self.links.append(link2)

					newNeuron = SNeuronGene(NeuronType.HIDDEN, newNeuronID, newDepth, newWidth)

					self.neurons.append(newNeuron)

	def createPhenotype(self, depth):
		# deletePhenotype()

		neurons = []

		for neuron in neurons:
			neurons.append(
				SNeuron(neuron.neuronType,
					neuron.neuronID,
					neuron.splitY,
					neuron.splitX,
					neuron.activationResponse))


		for link in links:
			if (links.enabled):

				fromNeuron = next((neuron 
					for neuron in enumerate(self.neurons) if neuron.ID == link.fromNeuron.ID), None)
				toNeuron = next((neuron 
					for neuron in enumerate(self.neurons) if neuron.ID == link.toNeuron.ID), None)

				tmpLink = SLink(link.weight,
					fromNeuron,
					toNeuron,
					link.recurrent)

				fromNeuron.linksOut.append(tmpLink)
				fromNeuron.linksIn.append(tmpLink)


		# phenotype = CNeuralNetwork()

		# return phenotype


def CNeuronNet:

	def __init__(self, neurons, depth):
		self.neurons = neurons
		self.depth = depth


	def update(inputs):
		outputs = []

		neuronIndex = 0

		# Set input neurons values 
		while(self.neurons[neuronIndex].neuronType == NeuronType.INPUT):
			self.neurons[neuronIndex].output = inputs[neuronIndex]

			neuronIndex += 1

		# Set bias
		self.neurons[neuronIndex].output = 1

		neuronIndex += 1

		for currentNeuron in self.neurons:
			neuronSum = 0.0

			for link in len(currentNeuron.linksIn):
				weight = link.weight

				neuronOutput = link.neuronIn.output

				neuronSum += weight * neuronOutput

			
			currentNeuron.output = sigmoid(neuronSum, currentNeuron.activationResponse)

			if (currentNeuron.neuronType == NeuronType.OUTPUT):
				outputs.append(currentNeuron.output)

		return outputs