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
	age = 0
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

	def adjustFitnesses(self):
		total = 0.0

		for member in self.members:
			fitness = member.fitness

			if (self.age < self.youngBonusAgeThreshold):
				fitness *= self.youngFitnessBonus

			if (self.age > self.oldAgeThreshold):
				fitness *= self.oldAgePenalty

			total += fitness

			adjustedFitness = fitness/len(self.members)

			member.adjustedFitness = adjustedFitness

class NEAT:
	genomes = []
	phenotypes = []
	species = []
	speciesNumber = 0	

	generation = 0
	currentGenomeID = 0

	numOfSweepers = None
	crossoverRate = 0.7
	# crossoverRate = 0.5
	# maxNumberOfNeuronsPermitted = 15
	maxNumberOfNeuronsPermitted = 10000
	
	newSpeciesTolerance = 0.5

	# chanceToAddNode = 0.5
	chanceToAddNode = 0.03
	numOfTriesToFindOldLink = 10
	
	chanceToAddLink = 0.05
	# chanceToAddLink = 1.0
	chanceToAddRecurrentLink = 0.00
	numOfTriesToFindLoopedLink = 15
	numOfTriesToAddLink = 20

	mutationRate = 0.8
	probabilityOfWeightReplaced = 0.01
	maxWeightPerturbation = 0.9

	activationMutationRate = 0.7
	maxActivationPerturbation = 0.8

	def __init__(self, numberOfGenomes, numOfInputs, numOfOutputs):
		self.numOfSweepers = numberOfGenomes

		newSpecies = CSpecies(self.speciesNumber)

		inputs = []
		# print("Creating input neurons:")
		for n in range(numOfInputs):
			print("\rCreating inputs neurons (" + str(n+1) + "/" + str(numOfInputs) + ")", end='')
			newInput = innovations.createNewNeuron(None, None, n, 0.0, NeuronType.INPUT)
			# print("neuron id:", newInput.ID)
			inputs.append(newInput)

		print("")

		biasInput = innovations.createNewNeuron(None, None, n, 0.0, NeuronType.BIAS)
		inputs.append(biasInput)

		outputs = []
		for n in range(numOfOutputs):
			print("\rCreating output neurons (" + str(n+1) + "/" + str(numOfOutputs) + ")", end='')
			newOutput = innovations.createNewNeuron(None, None, n, 1.0, NeuronType.OUTPUT)
			outputs.append(newOutput)

		print("")
		links = []
		# i = 0
		# for outputNeuron in outputs:
		# 	for inputNeuron in inputs:
		# 		print("\rCreating links (" + str(i+1) + "/" + str(len(outputs) * len(inputs)) + ")", end='')
		# 		# newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, 0.0)
		# 		if (inputNeuron.neuronType != NeuronType.BIAS):
		# 			newLink = innovations.createNewLink(inputNeuron, outputNeuron, True, 0.0)
		# 			links.append(newLink)
				
		# 		i += 1

		print("")
		inputs.extend(outputs)
		# print("created neurons:")
		# for neuron in inputs:
			# print("neuron ID:", neuron.ID)

		for i in range(self.numOfSweepers):
			newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
			self.genomes.append(newGenome)
			newSpecies.members.append(newGenome)

			self.currentGenomeID += 1

		for g in self.genomes:
			print("genome: ", g.genomeID)

		self.speciesNumber += 1
		self.species.append(newSpecies)

		self.phenotypes = self.epoch([0] * len(self.genomes))

	def crossover(self, mum, dad):
		best = None

		if (len(mum.links) < 1 or len(dad.links) < 1):
			print("mom or dad have no genes")
			best = random.choice([mum, dad])

			return CGenome(self.currentGenomeID, best.neurons, best.links, best.inputs, best.outputs)

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

			# print(currentMum.innovationID, currentDad.innovationID)

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
		babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

		# print("baby neurons:", len(babyNeurons))
		# for neuron in babyNeurons:
			# print("neuron:", neuron.ID, neuron.neuronType, neuron.splitY)

		self.currentGenomeID += 1
			
		return CGenome(self.currentGenomeID, babyNeurons, babyLinks, mum.inputs, mum.outputs)

	def epoch(self, fitnessScores):
		if (len(fitnessScores) != len(self.genomes)):
			print("Mismatch of scores/genomes size.")

		# TODO: ??
		# resetAndKill()

		print("Setting fitness scores")
		for index, genome in enumerate(self.genomes):
			print("id: ", genome.genomeID)
			print("fitenss score:", fitnessScores[index])
			genome.fitness = fitnessScores[index]

		# TODO: ??
		# sortAndRecord()

		# TODO: ??
		# speciateAndCalculateSpawnLevels()

		# self.species = []

		for genome in self.genomes:
			speciesMatched = False
			for s in self.species:
				distance = genome.calculateCompatibilityDistance(s.leader())
				print("distance: ", distance)
				if (distance < self.newSpeciesTolerance):
					print("Adding member to species " + str(s.ID))
					s.members.append(genome)
					speciesMatched = True

				# print("Distance: " + str(distance))

			if (not speciesMatched):
				self.speciesNumber += 1
				newSpecies = CSpecies(self.speciesNumber)
				print("Creating new species " + str(newSpecies.ID))
				newSpecies.members.append(genome)
				self.species.append(newSpecies)

		print("Number of species: " + str(len(self.species)))
		for s in self.species:
			s.age += 1
			s.adjustFitnesses()


		for s in self.species:
			print("adjust fitnesses: ", [m.adjustedFitness for m in s.members])
			avgFitness = sum([m.adjustedFitness for m in s.members])/len(s.members)
			
			print("avg fitness:", avgFitness)
			if (avgFitness == 0.0):
				continue

			for member in s.members:
				print("add to spawn: ", member.adjustedFitness, avgFitness, member.adjustedFitness/avgFitness)
				s.numToSpawn += member.adjustedFitness/avgFitness

			print("num to spawn: ", s.numToSpawn)

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

								# print("parents:", len(g1.links), len(g2.links))
								if (g1.genomeID != g2.genomeID or 
									len(g1.links) < 1 or len(g2.links)) < 1:
										baby = self.crossover(g1, g2)
								else:
									baby = g1

								self.currentGenomeID += 1.0

								baby.genomeID = self.currentGenomeID

								if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
									baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

								# print("baby neurons: ", len(baby.neurons))
								# print("baby links: ", len(baby.links))
								baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
									self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)
								
								baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
									self.maxWeightPerturbation)

								# baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

							baby.links.sort()

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
