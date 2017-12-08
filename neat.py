import random
from random import randint

import math
from enum import Enum

from copy import copy

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

	avgFitness = 0.0
	numOfGensWithoutImprovement = 0

	def __init__(self, speciesID):
		self.ID = speciesID

	def __contains__(self, key):
		return key.ID in [m.ID for m in self.members]

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


			adjustedFitness = fitness/len(self.members)
			
			total += adjustedFitness

			member.adjustedFitness = adjustedFitness

		self.avgFitness = total/len(self.members)

class NEAT:
	genomes = []
	phenotypes = []
	species = []
	speciesNumber = 0	

	generation = 0
	numGensAllowNoImprovement = 20

	currentGenomeID = 0

	numOfSweepers = None
	crossoverRate = 0.7
	# crossoverRate = 0.5
	# maxNumberOfNeuronsPermitted = 15
	maxNumberOfNeuronsPermitted = 10000
	
	newSpeciesTolerance = 0.9

	# chanceToAddNode = 0.3
	chanceToAddNode = 0.03
	numOfTriesToFindOldLink = 10
	
	chanceToAddLink = 0.05
	# chanceToAddLink = 0.5
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

		# for i in range(self.numOfSweepers):
		newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
		self.genomes.append(newGenome)
		newSpecies.members.append(newGenome)

		self.currentGenomeID += 1

		# for g in self.genomes:
			# print("genome: ", g.ID)

		self.speciesNumber += 1
		self.species.append(newSpecies)

		self.phenotypes = self.epoch([0] * len(self.genomes))


	def epoch(self, fitnessScores):
		if (len(fitnessScores) != len(self.genomes)):
			print("Mismatch of scores/genomes size.")
			return

		# TODO: ??
		# resetAndKill()

		print("Number of species:", len(self.species))

		for s in self.species:
			s.members = [s.leader()]

		for index, genome in enumerate(self.genomes):
			genome.fitness = fitnessScores[index]

		# print("Fitness scores: ", [(g.ID, g.fitness) for g in self.genomes])

		# TODO: ??
		# sortAndRecord()

		# TODO: ??
		# speciateAndCalculateSpawnLevels()

		# self.species = []

		print("Total number of genomes: ", len(self.genomes))
		print("Total number of species: ", len(self.species))
		for genome in self.genomes:
			speciesMatched = False
			# print("-------------------------")
			# print("Genome:", genome.ID)
			for s in self.species:
				distance = genome.calculateCompatibilityDistance(s.leader())
				# print("Species:", s.ID, "| Nr. of Members:", len(s.members), "| Leader:", s.leader().ID, "| Distance:", distance)
				# print("Members: ", [m.ID for m in s.members])

				if (genome in s):
					speciesMatched = True
					break

				if (distance < self.newSpeciesTolerance):
					# print("Adding member to species " + str(s.ID))
					s.members.append(genome)
					speciesMatched = True
					break

				# print("Distance: " + str(distance))

			if (not speciesMatched):
				self.speciesNumber += 1
				newSpecies = CSpecies(self.speciesNumber)
				print("Creating new species " + str(newSpecies.ID))
				newSpecies.members.append(genome)
				# print("Adding genome", genome.ID)
				self.species.append(newSpecies)

		# print("Number of species: " + str(len(self.species)))
		for s in self.species:
			s.age += 1
			s.adjustFitnesses()


		for s in self.species:
			# print("fitnesses: ", [m.fitness for m in s.members])
			# print("adjusted: ", [m.adjustedFitness for m in s.members])
			avgFitness = sum([m.adjustedFitness for m in s.members])/len(s.members)
			
			print("Species:", s.ID, "| No improvement:", s.numOfGensWithoutImprovement)
			print("Avg:", avgFitness, "| Species avg:", s.avgFitness)
			if (avgFitness <= s.avgFitness):
				s.numOfGensWithoutImprovement += 1

				if (s.numOfGensWithoutImprovement == self.numGensAllowNoImprovement):
					print("Removing species", s.ID, "for lack of improvement")
					self.species.remove(s)
					continue
			else:
				s.numOfGensWithoutImprovement = 0

			# print("avg fitness:", avgFitness)
			if (avgFitness == 0.0):
				continue

			# toSpawn = 0
			# for member in s.members:
			# 	# print("add to spawn: ", 
			# 		# member.fitness, member.adjustedFitness, avgFitness, member.adjustedFitness/avgFitness)
			# 	toSpawn += member.adjustedFitness/avgFitness

			# s.numToSpawn = toSpawn
			# print("num to spawn: ", s.numToSpawn)

		newPop = []
				
		# import pdb; pdb.set_trace()  # breakpoint 79b5a3f7 //

		numSpawnedSoFar = 0
		for speciesMember in self.species:
			if (numSpawnedSoFar < self.numOfSweepers):
				numToSpawn = round(speciesMember.numToSpawn)
				chosenBestYet = False

				for i in range(numToSpawn):
					baby = None
					# print("spawning")

					if (not chosenBestYet):
						baby = copy(speciesMember.leader())
						chosenBestYet = True

					else:
						if (len(speciesMember.members) == 1):
							baby = copy(speciesMember.spawn())
						else:
							g1 = speciesMember.spawn()

							if (random.random() < self.crossoverRate):
								g2 = speciesMember.spawn()

								numOfAttempts = 5

								while((g1.ID == g2.ID) and (numOfAttempts > 0)):
									numOfAttempts -= 1
									g2 = speciesMember.spawn()

								if (g1.ID != g2.ID):
									baby = self.crossover(g1, g2)
								else:
									baby = copy(g1)
							else:
								baby = copy(g1)

						self.currentGenomeID += 1
						baby.ID = self.currentGenomeID

						for i in range(10):
							if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
								baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

						for i in range(10):
							baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
								self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)
						
						baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
							self.maxWeightPerturbation)

						# baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)

					baby.links.sort()

					# print("Adding new baby:", baby)
					newPop.append(baby)
					# print("New pop:", [g.ID for g in newPop])
					numSpawnedSoFar += 1

					# if (numSpawnedSoFar == self.numOfSweepers):
						# numToSpawn = 0
						# break

		# print("newpop:", len(newPop))
		if (numSpawnedSoFar < self.numOfSweepers):
			requiredNumberOfSpawns = self.numOfSweepers - numSpawnedSoFar

				# for i in requiredNumberOfSpawns:
					# newPop.append(TournamentSelection())

		# print("New pop:", [g.ID for g in newPop])
		self.genomes = newPop
		# print("New pop:", [g.ID for g in self.genomes])
		
		newPhenotypes = []

		for genome in self.genomes:
			# depth = calculateNetDepth(genome

			depth = len(set(n.splitY for n in genome.neurons))
			phenotype = genome.createPhenotype(depth)

			newPhenotypes.append(phenotype)


		self.generation += 1

		return newPhenotypes

	def crossover(self, mum, dad):
		best = None

		# if (len(mum.links) < 1 or len(dad.links) < 1):
		# 	print("mom or dad have no genes")
		# 	best = random.choice([mum, dad])

		# 	self.currentGenomeID += 1
		# 	return CGenome(self.currentGenomeID, best.neurons, best.links, best.inputs, best.outputs)

		if (mum.fitness == dad.fitness):
			if (len(mum.links) == len(dad.links)):
				best = random.choice([mum, dad])
			else:
				best = mum if len(mum.links) < len(dad.links) else dad
		else:
			best = mum if mum.fitness > dad.fitness else dad


		# print("best is ", "mum" if best == mum else "dad")

		babyNeurons = [n for n in mum.neurons 
			if (n.neuronType in [NeuronType.INPUT, NeuronType.BIAS, NeuronType.OUTPUT])]
		babyLinks = []

		# print("parent neurons: ", len(mum.neurons), len(dad.neurons))
		# print("parent links: ", len(mum.links), len(dad.links))
		selectedLink = None
		mumIt = dadIt = 0
		# while(not (mumIt == len(mum.links) and dadIt == len(dad.links))):
		while(mumIt < len(mum.links) or dadIt < len(dad.links)):
			
			if mumIt < len(mum.links):
				currentMum = mum.links[mumIt]

			if dadIt < len(dad.links):
				currentDad = dad.links[dadIt]


			if (mumIt == (len(mum.links)) and (dadIt != len(dad.links))):
				if (best == dad):
					selectedLink = currentDad

				dadIt += 1

			elif (dadIt == (len(dad.links)) and (mumIt != len(mum.links))):
				if (best == mum):
					selectedLink = currentMum

				mumIt += 1

			elif (currentDad.innovationID == currentMum.innovationID):
				selectedLink = random.choice([currentMum, currentDad])

				mumIt += 1
				dadIt += 1

			elif (currentMum.innovationID < currentDad.innovationID):
				if (best == mum):
					selectedLink = currentMum

				mumIt += 1

			elif (currentDad.innovationID < currentMum.innovationID):
				if (best == dad):
					selectedLink = currentDad

				dadIt += 1

			if (selectedLink != None):
				if (len(babyLinks) == 0):
					babyLinks.append(selectedLink)
				else:
					if (babyLinks[-1].innovationID != selectedLink.innovationID):
						babyLinks.append(selectedLink)

				if (selectedLink.fromNeuron not in babyNeurons):
					babyNeurons.append(selectedLink.fromNeuron)

				if (selectedLink.toNeuron not in babyNeurons):
					babyNeurons.append(selectedLink.toNeuron)

		babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

		self.currentGenomeID += 1

		if (len(babyLinks) < 1 or len(babyNeurons) < 1):
			best = random.choice([mum, dad])
			babyNeurons = best.neurons
			babyLinks = best.links

		return CGenome(self.currentGenomeID, babyNeurons, babyLinks, mum.inputs, mum.outputs)