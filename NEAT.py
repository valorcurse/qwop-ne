import random
from random import randint

import math
from enum import Enum

import genes

class CSpecies:
	members = []
	species = []
	age = 0

	# TODO: Find out correct values
	youngBonusAgeThreshold = 5
	youngFitnessBonus = 1.5
	oldAgeThreshold = 15
	oldAgePenalty = 0.5

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
	generation = 0
	currentGenomeID = 0

	numOfSweepers = 10
	crossoverRate = 0.5
	maxNumberOfNeuronsPermitted = 15
	
	chanceToAddNode = 0.6
	numOfTriesToFindOldLink = 10
	
	chanceToAddLink = 0.4
	chanceToAddRecurrentLink = 0.1
	numOfTriesToFindLoopedLink = 15
	numOfTriesToAddLink = 20

	mutationRate = 0.7
	probabilityOfWeightReplaced = 0.3
	maxWeightPerturbation = 0.8

	activationMutationRate = 0.7
	maxActivationPerturbation = 0.8

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
		babyGenes = []

		neuronsList = []

		selectedGene = None
		while(mumIt < len(mum.links) and dadIt < len(dad.links)):
			currentMum = mum.links[mumIt]
			currentDad = dad.links[dadIt]

			if (mumIt == (len(mum.links) - 1) and (dadIt < len(dad.links))):
				if (best == dad):
					selectedGene = currentDad
				dadIt += 1

			elif (dadIt == (len(dad.links) - 1) and (mumIt < len(mum.links))):
				if (best == mum):
					selectedGene = currentMum
				mumIt += 1

			elif (currentMum.innovationID < currentDad.innovationID):
				if (best == mum):
					selectedGene = currentMum

				mumIt += 1

			elif (currentDad.innovationID < currentMum.innovationID):
				if (best == dad):
					selectedGene = currentDad

				dadIt += 1

			elif (currentDad.innovationID == currentMum.innovationID):
				selectedGene = random.choice([currentMum, currentDad])

				mumIt += 1
				dadIt += 1

			
			if (len(babyGenes) == 0):
				babyGenes.append(selectedGene)
			else:
				if (babyGenes[-1].innovationID != selectedGene.innovationID):
					babyGenes.append(selectedGene)


			if len([neuron for neuron in babyNeurons if neuron.innovationID == selectedGene.fromNeuron.innovationID]) <= 0:
				babyNeurons.append(selectedGene.fromNeuron)

			if len([neuron for neuron in babyNeurons if neuron.innovationID == selectedGene.toNeuron.innovationID]) <= 0:
				babyNeurons.append(selectedGene.toNeuron)

			babyNeurons.sort(key=lambda x: x.innovationID, reverse=True)

			self.currentGenomeID += 1
			
			return CGenome(self.currentGenomeID, babyNeurons, babyGenes, mum.inputs, mum.outputs)

	def epoch(self, fitnessScores):
		if (len(fitnessScores) != len(genomes)):
			print("Mismatch of scores/genomes size.")

		# TODO: ??
		# resetAndKill()

		for index, genome in self.genomes:
			genome.fitness = fitnessScores[index]

		# TODO: ??
		# sortAndRecord()

		# TODO: ??
		# speciateAndCalculateSpawnLevels()

		newPop = []
		numSpawnedSoFar = 0

		baby = None

		for speciesMember in species:

			if (numSpawnedSoFar < self.numOfSweepers):

				numToSpawn = math.ceil(speciesMember.numToSpawn)
				chosenBestYet = False

				for i in numToSpawn:

					if (not chosenBestYet):
						baby = speciesMember.leader

						chosenBestYet = True

					else:
						if (speciesMember.numOfMembers == 1):
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

								currentGenomeID += 1

								baby.genomeID = currentGenomeID

								if (len(baby.neurons) < self.maxNumberOfNeuronsPermitted):
									baby.addNeuron(self.chanceToAddNode, self.numOfTriesToFindOldLink)

								baby.addLink(self.chanceToAddLink, self.chanceToAddRecurrentLink,
									self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)

								baby.mutateWeights(self.mutationRate, self.probabilityOfWeightReplaced,
									self. maxWeightPerturbation)

								baby.mutateActivationResponse(self.activationMutationRate, self.maxActivationPerturbation)


							baby.links.sort(key=lambda x: x.innovationID, reverse=True)

							newPop.append(baby)

							numSpawnedSoFar += 1

							if (numSpawnedSoFar == self.numOfSweepers):
								numToSpawn = 0

		if (numSpawnedSoFar < self.numOfSweepers):
			requiredNumberOfSpawns = numOfSweepers - numSpawnedSoFar

			# for i in requiredNumberOfSpawns:
				# newPop.append(TournamentSelection())

		self.genomes = newPop

		newPhenotypes = []

		# for gen in self.genomes:
			# depth = calculateNetDepth()
			# phenotype = createPhenotype()

			# newPhenotypes.append(phenotype)


		generation += 1

		return newPhenotypes

class SLink:
	
	def __init__():
		self.neuronIn = None
		self.neuronOut = None

		self.weight = 0

		self.recurrent = False

