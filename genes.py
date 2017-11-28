import random
from random import randint

import math
from enum import Enum

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

class Innovations:
	listOfInnovations = []

	currentNeuronID = 0

	def createNewLinkInnovation(self, fromNeuron, toNeuron):
		newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromNeuron, toNeuron, -1, NeuronType.LINK)
		self.listOfInnovations.append(newInnovation)

		return len(self.listOfInnovations) - 1;

	def createNewLink(self, fromNeuron, toNeuron, enabled, weight, recurrent=False):
		ID = self.createNewLinkInnovation(fromNeuron, toNeuron)
		return SLinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

	def createNewNeuronInnovation(self, fromNeuron, toNeuron):
		neurons = [neuron for neuron in self.listOfInnovations if neuron.innovationType == InnovationType.NEURON]
		newNeuronID = len(neurons)
		newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations), -1, -1, newNeuronID, NeuronType.HIDDEN)
		self.listOfInnovations.append(newInnovation)

		return len(self.listOfInnovations) - 1;

	def createNewNeuron(self, fromNeuron, toNeuron, x, y, neuronType, recurrent = False):
		ID = self.createNewNeuronInnovation(fromNeuron, toNeuron)
		self.currentNeuronID += 1
		return SNeuronGene(neuronType, self.currentNeuronID, x, y, ID, recurrent)

	def checkInnovation(self, start, end, innovationType):
		matched = [index for index, innovation in self.listOfInnovations if ((innovation.start == start) and (innovation.end == end) and (innovation.innovationType == innovationType))]

		return matched

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
	def __init__(self, neuronType, ID, x, y, innovationID, recurrent = False):
		self.ID = ID
		self.neuronType = neuronType
		self.recurrent = recurrent
		self.activationResponse = None
		self.splitX = x
		self.splitY = y

		self.innovationID = None

		self.numInputs = 0
		self.numOutputs = 0

		self.neurons = []
		self.links = []

class SLink:
	
	def __init__(self, neuronIn, neuronOut, weight, recurrent = False):
		self.neuronIn = neuronIn
		self.neuronOut = neuronOut

		self.weight = weight

		self.recurrent = recurrent

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

	fitness = 0

	def __init__(self, ID, neurons, links, inputs, outputs):
		self.genomeID = ID
		self.neurons = neurons
		self.links = links
		self.inputs = inputs
		self.outputs = outputs

		print("init: Links lengths", len(self.links))

	def calculateCompatibilityDistance(self, otherGenome):
		g1 = g2 = 0
		numExcess = 0
		numMatched = 0
		numDisjointed = 0

		weightDifference = 0

		while ((g1 < len(self.links)) or (g2 < len(otherGenome.links))):
			if (g1 == len(self.links) - 1):
				g2 += 1
				numExcess += 1

				continue

			if (g2 == len(otherGenome.links)):
				g1 += 1
				numExcess += 1

				continue

			id1 = self.links[g1].innovationID
			id2 = otherGenome.links[g2].innovationID

			if (id1 == id2):
				g1 += 1
				g2 += 1
				numMatched += 1

				weightDifference += math.fabs(self.links[g1].weight - otherGenome.links[g2].weight)

			if (id1 < id2):
				numDisjointed += 1
				g1 += 1

			if (id1 > id2):
				numDisjointed += 1
				g2 += 1

		longest = len(otherGenome.links) if len(otherGenome.links) > len(self.links) else len(self.links)

		disjoint = 1.0
		excess = 1.0
		matched = 0.4

		return ((excess * numExcess / longest) +
			(disjoint * numDisjointed / longest) +
			(matched * weightDifference / numMatched))

	def addLink(self, mutationRate, chanceOfLooped, triesToFindLoop, triesToAddLink):

		if (random.random() > mutationRate):
			return

		neuron1 = -1
		neuron2 = -1
		recurrent = False

		if (random.random() < chanceOfLooped):
			loopFound = False
			while(triesToFindLoop and not loopFound):
				neuronPosition = randint(numInputs + 1, len(self.neurons) - 1)

				loopNeuron = self.neurons[neuronPosition]
				if (not loopNeuron.recurrent or 
					loopNeuron.neuronType != NeuronType.BIAS or
					loopNeuron.neuronType != NeuronType.INPUT):

						neuron1 = neuron2 = loopNeuron.ID

						recurrent = loopNeuron.recurrent = True

						loopFound = True
		else:
			loopFound = False
			while(triesToAddLink and not loopFound):
				print("addLink neurons:", len(self.neurons))
				neuron1 = self.neurons[randint(0, len(self.neurons) - 1)]
				neuron2 = self.neurons[randint(1, len(self.neurons) - 1)]

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

	def addNeuron(self, mutationRate, triesToFindOldLink):

		print("Links length", len(self.links))

		if (random.random() > mutationRate):
			return

		done = False
		chosenLink = 0

		sizeThreshold = self.inputs + self.outputs + 5

		if (len(self.links) < sizeThreshold):

			loopFound = False
			while (triesToFindOldLink and not loopFound):
				# Genes might not be the same as links
				maxRand = len(self.links) - 1 - math.sqrt(len(self.links))
				print("max rand")
				print(maxRand)
				chosenLink = self.links[randint(0, maxRand)]

				fromNeuron = chosenLink.fromNeuron

				if (chosenLink.enabled and
					not chosenLink.recurrent and
					self.neurons.index(fromNeuron).neuronType != NeuronType.BIAS):

						done = loopFound = True

			if (not done):
				return

		else:

			while(not done):
				chosenLink = self.links[randint(0, len(self.links) - 1)]
				fromNeuron = chosenLink.fromNeuron
				print(fromNeuron, chosenLink.fromNeuron)
				if (chosenLink.enabled and
					not chosenLink.recurrent and
					self.neurons.index(fromNeuron).neuronType is not NeuronType.BIAS):
						done = True

				chosenLink.enabled = True

				originalWeight = chosenLink.weight
				
				fromNeuron = chosenLink.fromNeuron
				toNeuron = chosenLink.toNeuron

				newDepth = (self.neurons.index().splitY +
					self.neurons.index(chosenLink.toNeuron).splitY) / 2

				newWidth = (self.neurons.index(chosenLink.fromNeuron).splitX +
					self.neurons.index(chosenLink.toNeuron).splitX) / 2

				ID = innovations.checkInnovation(fromNeuron, toNeuron, InnovationType.NEURON)

				if (ID >= 0):
					neuronID = innovations.getInnovation(ID).neuronID

					if (self.neurons.index(neuronID) >= 0):
						ID = -1


				if (ID < 0):
					newNeuronID = innovations.createNewNeuronInnovation(fromNeuron, toNeuron, newWidth, newDepth)

					self.neurons.append(SNeuronGene(NeuronType.HIDDEN,
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


		for link in self.links:
			if (link.enabled):

				fromNeuron = next((neuron 
					for neuron in (neurons) if (neuron.ID == link.fromNeuron.ID)), None)
				toNeuron = next((neuron 
					for neuron in (neurons) if (neuron.ID == link.toNeuron.ID)), None)

				tmpLink = SLink(link.weight,
					fromNeuron,
					toNeuron,
					link.recurrent)

				fromNeuron.linksOut.append(tmpLink)
				fromNeuron.linksIn.append(tmpLink)


		return CNeuralNetwork(neurons, depth)

class CNeuronNet:

	def __init__(self, neurons, depth):
		self.neurons = neurons
		self.depth = depth


	def update(self, inputs):
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