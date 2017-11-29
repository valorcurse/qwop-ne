import random
from random import randint

import math
from enum import Enum

from operator import attrgetter

import genes
from genes import NeuronType
from genes import CGenome
from genes import innovations

global innovations

class CSpecies:
	leader = None
	members = []
	ID = None
	age = None
	numToSpawn = 5

	# TODO: Find out correct values
	youngBonusAgeThreshold = 5
	youngFitnessBonus = 1.5
	oldAgeThreshold = 15
	oldAgePenalty = 0.5

	def __init__(self, speciesID):
		self.ID = speciesID

	def leader(self):
		return max(self.members, key=attrgetter('fitness'))

	def spawn(self):
		return random.choice(self.members)

	def adjustFitness():
		total = 0.0

		for member in members:
			fitness = member.fitness

			if (age <  youngBonusAgeThreshold):
				fitness *= youngFitnessBonus

			if (age > oldAgeThreshold):
				fitness *= oldAgePenalty

			total += fitness

			adjustedFitness = fitness/len(members)

			member.adjustedFitness = adjustedFitness

class NEAT:
	genomes = []
	phenotypes = []
	species = []
	speciesNumber = 0	

	generation = 0
	currentGenomeID = 0

	numOfSweepers = None
	crossoverRate = 0.5
	maxNumberOfNeuronsPermitted = 15
	
	chanceToAddNode = 0.5
	numOfTriesToFindOldLink = 10
	
	chanceToAddLink = 0.03
	chanceToAddRecurrentLink = 0.01
	numOfTriesToFindLoopedLink = 15
	numOfTriesToAddLink = 20

	mutationRate = 0.8
	probabilityOfWeightReplaced = 0.1
	maxWeightPerturbation = 0.9

	activationMutationRate = 0.7
	maxActivationPerturbation = 0.8

	def __init__(self, numberOfGenomes, numOfInputs, numOfOutputs):
		self.numOfSweepers = numberOfGenomes

		newSpecies = CSpecies(self.speciesNumber)

		inputs = []
		for n in range(numOfInputs):
			print("\rCreating inputs neurons (" + str(n+1) + "/" + str(numOfInputs) + ")", end='')
			newInput = innovations.createNewNeuron(-1, -1, n, 0, NeuronType.INPUT)
			inputs.append(newInput)

		print("")
		outputs = []
		for n in range(numOfOutputs):
			print("\rCreating output neurons (" + str(n+1) + "/" + str(numOfOutputs) + ")", end='')
			newOutput = innovations.createNewNeuron(-1, -1, n, 0, NeuronType.OUTPUT)
			outputs.append(newOutput)

		print("")
		links = []
		i = 0
		for outputNeuron in outputs:
			for inputNeuron in inputs:
				print("\rCreating links (" + str(i+1) + "/" + str(len(outputs) * len(inputs)) + ")", end='')
				# newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, 0.0)
				newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, random.random())
				links.append(newLink)
				i += 1

		print("")
		inputs.extend(outputs)
		for i in range(self.numOfSweepers):
			newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
			self.genomes.append(newGenome)
			newSpecies.members.append(newGenome)

			self.currentGenomeID += 1

		self.speciesNumber += 1
		self.species.append(newSpecies)

		self.phenotypes = self.epoch([0] * len(self.genomes))

	def crossover(self, mum, dad):
		best = None

		if (mum.fitness == dad.fitness):
			if (len(mum.links) == len(dad.links)):
				best = random.choice([mum, dad])
			else:
				best = mum if len(mum.links) < len(dad.links) else dad
		else:
			best = mum if mum.fitness > dad.fitness else dad


		mumIt = dadIt = 0

		babyNeurons = []
		babyLinks = []

		selectedLink = None
		while(not (mumIt == len(mum.links)) and not (dadIt == len(dad.links))):
			currentMum = mum.links[mumIt]
			currentDad = dad.links[dadIt]

			if (mumIt == (len(mum.links) - 1) and (dadIt < len(dad.links))):
				if (best == dad):
					selectedLink = currentDad
				dadIt += 1

			elif (dadIt == (len(dad.links) - 1) and (mumIt < len(mum.links))):
				if (best == mum):
					selectedLink = currentMum
				mumIt += 1

			elif (currentMum.innovationID < currentDad.innovationID):
				if (best == mum):
					selectedLink = currentMum

				mumIt += 1

			elif (currentDad.innovationID < currentMum.innovationID):
				if (best == dad):
					selectedLink = currentDad

				dadIt += 1

			elif (currentDad.innovationID == currentMum.innovationID):
				selectedLink = random.choice([currentMum, currentDad])

				mumIt += 1
				dadIt += 1

			
			if (len(babyLinks) == 0):
				babyLinks.append(selectedLink)
			else:
				if (babyLinks[-1].innovationID != selectedLink.innovationID):
					babyLinks.append(selectedLink)

			if (selectedLink.fromNeuron not in babyNeurons):
				babyNeurons.append(selectedLink.fromNeuron)

			if (selectedLink.toNeuron not in babyNeurons):
				babyNeurons.append(selectedLink.toNeuron)

		# print("babyneurons:", len(babyNeurons))
		babyNeurons.sort(key=lambda x: x.ID, reverse=True)

		self.currentGenomeID += 1
			
		return CGenome(self.currentGenomeID, babyNeurons, babyLinks, mum.inputs, mum.outputs)

	def epoch(self, fitnessScores):
		if (len(fitnessScores) != len(self.genomes)):
			print("Mismatch of scores/genomes size.")

		# TODO: ??
		# resetAndKill()

		print("Setting fitness scores")
		for index, genome in enumerate(self.genomes):
			genome.fitness = fitnessScores[index]

		# TODO: ??
		# sortAndRecord()

		# TODO: ??
		# speciateAndCalculateSpawnLevels()

		# for genome in self.genomes:
		# 	for species in self.species:
		# 		distance = genome.calculateCompatibilityDistance(species.members[0])
		# 		print("Distance: " + str(distance))

		newPop = []
		numSpawnedSoFar = 0

		baby = None

		for speciesMember in self.species:

			if (numSpawnedSoFar < self.numOfSweepers):

				numToSpawn = math.ceil(speciesMember.numToSpawn)
				chosenBestYet = False

				for i in range(numToSpawn):

					if (not chosenBestYet):
						baby = speciesMember.leader()

						chosenBestYet = True

					else:
						if (len(speciesMember.members) == 1):
							baby = speciesMember.spawn()
						else:
							g1 = speciesMember.spawn()

							if (random.random() < self.crossoverRate):
								g2 = speciesMember.spawn()

								numOfAttempts = 5

								while((g1.genomeID == g2.genomeID) and numOfAttempts):
									numOfAttempts -= 1

									g2 = speciesMember.spawn()

								if (g1.genomeID != g2.genomeID):
									baby = self.crossover(g1, g2)
								else:
									baby = g1

								self.currentGenomeID += 1

								baby.genomeID = self.currentGenomeID

								if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
									baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

								baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
									self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

								for link in baby.links:
									if (random.random() > self.mutationRate):
										link.weight = random.random()
								# baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
								# 	self. maxWeightPerturbation)

								# baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)


							baby.links.sort(key=lambda x: x.innovationID, reverse=True)

							newPop.append(baby)

							numSpawnedSoFar += 1

							if (numSpawnedSoFar == self.numOfSweepers):
								numToSpawn = 0

		# print("newpop:", len(newPop))
		if (numSpawnedSoFar < self.numOfSweepers):
			requiredNumberOfSpawns = self.numOfSweepers - numSpawnedSoFar

			# for i in requiredNumberOfSpawns:
				# newPop.append(TournamentSelection())

		self.genomes = newPop

		newPhenotypes = []

		for genome in self.genomes:
			# depth = calculateNetDepth(genome)
			# print("neurons:", genome.neurons)
			depth = len(set(n.splitY for n in genome.neurons))
			phenotype = genome.createPhenotype(depth)

			newPhenotypes.append(phenotype)


		self.generation += 1

		return newPhenotypes