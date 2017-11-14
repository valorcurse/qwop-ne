import random
from random import randint

import math

innovations = []

class SLinkGene:

	def __init__(self, inNeuron, outNeuron, enable, tag, weight, recurrent=False):
		self.fromNeuron = inNeuron
		self.toNeuron = outNeuron

		self.weight = weight

		self.enabled = enabled

		self.recurrent = recurrent

		self.innovationID = tag

	def __lt__(self, other):
		return self.innovationID < other.innovationID

class NeuronType(Enum):
	INPUT = 0
	HIDDEN = 1
	BIAS = 2
	OUTPUT = 3
	LINK = 4

class SNeuronGene:
	
	self.numInputs = 0
	self.numOutputs = 0

	self.neurons = []
	self.links = []

	def __init__(neuronType, ID, x, y, recurrent = False):
		self.ID = ID
		self.neuronType = neuronType
		self.recurrent = recurrent
		self.activationResponse = None
		self.splitX = x
		self.splitY = y

class CGenome:

	def __init__(ID, neurons, genes, inputs, outputs):
		self.genomeID = ID
		self.neurons = neurons
		self.links = genes

	def addLink(self, mutationRate, chanceOfLooped, innovation, 
		triesToFindLoop, triesToAddLink):

		if (random.random() > mutationRate)
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
					loopNeuron.neuronType not NeuronType.BIAS or
					loopNeuron.neuronType not NeuronType.INPUT):

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

				if (# link not duplicate
					or neuron1.ID is neuron2.ID):

						loopFound = True

				else:
					neuron1.ID = neuron2.ID = -1


		if (neuron1.ID < 0 or neuron2.ID < 0):
			return

		# TBA later
		# ID = checkInnovation(...)
		ID = -1
		
		if (neuron1.splitY > neuron2.splitY)
			recurrent = True

		if (ID < 0):
			# new_link??
			innovation.createNewInnovation(neuron1, neuron2, NeuronType.LINK)

			ID = innovation.nextNumber() - 1

			randomClamped = random.random() - random.random()
			newGene = SLinkGene(neuron1, neuron2, true, id, randomClamped, recurrent)
			links.append(newGene)

		else:
			newGene = SLinkGene(neuron1, neuron2, true, id, randomClamped, recurrent)
			links.append(newGene)


	def addNeuron(self, mutationRate, innovations, triesToFindOldLink):

		if (random.random() > mutationRate)
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
					neurons.index(fromNeuron).neuronType is not NeuronType.BIAS):

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

				# TODO: Check for innovation here also
				ID = -1

				if (ID >= 0):
					# TODO: Get ID from innovations
					neuronID = -1

					if (neurons.index(neuronID) >= 0):
						ID = -1


				if (ID < 0):
					# TODO : function to create new innovation
					# newNeuronID = createNewInnovation()
					newNeuronID = -1

					neurons.append(SNeuronGene(NeuronType.HIDDEN,
						newNeuronID,
						newDepth,
						newWidth))

					# TODO: get next number from innovations
					idLink1 = innovations.nextNumber()

					innovations.createNewInnovation(fromNeuron, newNeuronID, NeuronType.LINK)

					link1 = SLinkGene(
						fromNeuron,
						newNeuronID,
						true,
						idLink1,
						1.0)


					idLink2 = innovations.nextNumber()
					innovations.createNewInnovation(newNeuronID, toNeuron, NeuronType.LINK)
					link2 = SLinkGene(
						newNeuronID,
						toNeuron,
						true,
						idLink2,
						originalWeight)

					self.links.push(link2)

				else:

					# TODO: Whole innovations shizzle
					newNeuronID = innovations.getNeuronID(ID)

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