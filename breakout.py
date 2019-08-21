from typing import List, Set, Dict, Tuple, Optional, Any

import neat.hyperneat as hn
from neat.genes import Genome
from neat.phenotypes import Phenotype
from neat.speciatedPopulation import SpeciesConfiguration, SpeciesUpdate

from visualize import Visualize

from multiprocessing.dummy import Pool as ThreadPool 

from prettytable import PrettyTable
import numpy as np
import scipy as sp
import math

import os
import sys
import dill
import _pickle as pickle
import argparse


from copy import deepcopy

import gym

environment = "Breakout-v0"

dimensions: int = 8
k: int = 15
max_frames: int = 1000
maxMapDistance: float = 300.0

space_range: int = 100
# Farthest distance between two points in specified number of dimensions
farthestDistance: float = np.sqrt(np.power((space_range*2), 2)*dimensions)
# Sparseness threshold as percentage of farthest distance between 2 points
# p_threshold: float = farthestDistance*0.03
p_threshold: float = 5.0

def testOrganism(phenotype: Phenotype, displayEnv: Any = None) -> Dict[str, Any]:
    env = displayEnv if displayEnv is not None else gym.make(environment)
    observation = env.reset()
    
    rewardSoFar: float = 0.0
    distanceSoFar: float = 0.0
    
    actionsDone = np.zeros(8)
    behavior = np.zeros(4)

    sparseness: float = 0
    nrOfSteps: int = 0
    totalSpeed: float = 0

    previousDistance: float = 0.0
    rewardStagnation: int = 0 

    done = False
    while not done:
# 
        action = phenotype.update(observation)
        
        state, reward, done, info = env.step(action)
        
        if (displayEnv is not None):
            env.render()

        print(state.shape)        
    
    return {
    #     # "behavior": [actionsDone],
    #     "behavior": behavior,
    #     # "speed": totalSpeed,
    #     # "nrOfSteps": nrOfSteps,
    #     "distanceTraveled": distanceSoFar,
    #     # "fitness": rewardSoFar
        "fitness": fitness
    }



if __name__ == '__main__':

    sys.setrecursionlimit(10000)

    # Save generations to file
    saveFolder = "bipedal"
    saveDirectory = "saves/" + saveFolder
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    parser = argparse.ArgumentParser()
    parser.add_argument("--load")

    args = parser.parse_args()

    env = gym.make(environment)

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    print("Creating hyperneat object")
    pop_config = SpeciesConfiguration(300, inputs, outputs)
    hyperneat = hn.HyperNEAT(pop_config)

    fitnesses: List[float] = []
    phenotypes: List[Phenotype] = []

    # randomPop = hyperneat.neat.population.randomInitialization()
    # for i, genome in enumerate(randomPop):
    #     print("\rInitializing start population ("+ str(i) +"/"+ str(len(randomPop)) +")", end='')
    #     fitnesses.append(testOrganism(hyperneat.createSubstrate(genome).createPhenotype())["fitness"])
    # phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))

    displayEnv = gym.make(environment)
    displayEnv.render()
    pool = ThreadPool(4) 

    highestFitness: float = 0.0
    highestDistance: float = 0.0

    while True:
        
        fitness = 0.0
        distance = 0.0
        # behavior = np.zeros(4)

        runs = 5
        # candidate = hyperneat.createSubstrate(cppn)
        
        # Parallel
        # output = pool.map(testOrganism, [candidate.createPhenotype()]*5)
        # fitness = max([l["fitness"] for l in output])
        # distance = max([l["distanceTraveled"] for l in output])
        # behavior = np.array([l["behavior"] for l in output])
        # behavior = np.add.reduce(behavior)/runs
        
        for phenotype in phenotypes:
            Visualize().update(phenotype)
            output = testOrganism(phenotype, displayEnv)
        
            fitness = max(fitness, output["fitness"])
            fitnesses.append(fitness)

            if fitness > highestFitness:
                print("New highest fitness: %f"%(fitness))
                # Visualize().update(cppn.createPhenotype())
                Visualize().update(phenotype)
                testOrganism(phenotype, displayEnv)
                highestFitness = fitness

        table = PrettyTable(["ID", "age", "members", "max fitness", "adj. fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. bias", "avg. compat.", "to spawn"])
        for s in hyperneat.neat.population.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                int(max([m.fitness for m in s.members])),                   # Max fitness
                "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
                # "{}".format(max([m.uniqueKeysPressed for m in s.members])), # Average unique keys
                s.generationsWithoutImprovement,                            # Stagnation
                int(np.mean([len(m.neurons)-1894 for m in s.members])),     # Neurons
                np.mean([len(m.links) for m in s.members]),                 # Links
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))

    env.close()