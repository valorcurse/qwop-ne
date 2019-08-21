from typing import List, Set, Dict, Tuple, Optional, Any

from neat.hyperneat import HyperNEAT
from neat.genes import Genome
from neat.phenotypes import CNeuralNet
from neat.mapelites import MapElitesConfiguration, Feature

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

def testOrganism(phenotype: CNeuralNet, displayEnv: Any = None) -> Dict[str, Any]:
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

    # substrateLayout: List[List[int]] = [
    #     "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i0", "i11", "i12", "i13", "i14",
    # ]

    hyperneat = None
    novelty_map = None
    if (args.load):
        print("Loading hyperneat object from file")
        with open("saves/" + args.load, "rb") as load:
            hyperneat, novelty_map = pickle.load(load)
            hyperneat.populationSize = 150
            # hyperneat.milestone = 157.0
    else:
        print("Creating hyperneat object")
        popConfig = MapElitesConfiguration(8, [
                Feature("hull_angular", 0, 2*math.pi), 
                # Feature("hull_angularVelocity", -1.0, 1.0), 
                Feature("vel_x", -1.0, 1.0),
                # Feature("vel_y", -1.0, 1.0),
                # Feature("hip_joint_1_angle", -1.0, 1.0),
                # Feature("hip_joint_1_speed", -1.0, 1.0),
                # Feature("knee_joint_1_angle", -1.0, 1.0),
                # Feature("knee_joint_1_speed", -1.0, 1.0),
                Feature("leg_1_ground_contact_flag", 0.0, 1.0),
                # Feature("hip_joint_2_angle", -1.0, 1.0),
                # Feature("hip_joint_2_speed", -1.0, 1.0),
                # Feature("knee_joint_2_angle", -1.0, 1.0),
                # Feature("knee_joint_2_speed", -1.0, 1.0),
                Feature("leg_2_ground_contact_flag", 0.0, 1.0)
            ])
        hyperneat = HyperNEAT(500, inputs, outputs, popConfig)
        # novelty_map = np.empty((0, 14), float)

    # randomPop = hyperneat.neat.population.randomInitialization()
    # for i, genome in enumerate(randomPop):
    #     print("\rInitializing start population ("+ str(i) +"/"+ str(len(randomPop)) +")", end='')
    #     output = testOrganism(hyperneat.createSubstrate(genome).createPhenotype())
    #     hyperneat.neat.updateCandidate(genome, output["fitness"], output["behavior"])

    displayEnv = gym.make(environment)
    displayEnv.render()
    pool = ThreadPool(4) 

    highestFitness: float = 0.0
    highestDistance: float = 0.0

    while True:
        cppn = hyperneat.getCandidate()
        
        fitness = 0.0
        distance = 0.0
        behavior = np.zeros(4)
        runs = 5
        candidate = hyperneat.createSubstrate(cppn)
        
        # Parallel
        # output = pool.map(testOrganism, [candidate.createPhenotype()]*5)
        # fitness = max([l["fitness"] for l in output])
        # distance = max([l["distanceTraveled"] for l in output])
        # behavior = np.array([l["behavior"] for l in output])
        # behavior = np.add.reduce(behavior)/runs
        
        for _ in range(runs):
            phenotype = hyperneat.createSubstrate(cppn).createPhenotype()
            Visualize().update(phenotype)
            output = testOrganism(phenotype, displayEnv)
        
            fitness = max(fitness, output["fitness"])
            distance = max(distance, output["distanceTraveled"])
            behavior += output["behavior"]

        # Visualize().update(candidate.createPhenotype())
        # testOrganism(candidate.createPhenotype(), displayEnv)
        # behavior /= runs

        if distance > highestDistance:
            highestDistance = distance

        if fitness > highestFitness:
            print("New highest fitness: %f"%(fitness))
            phenotype = hyperneat.createSubstrate(cppn).createPhenotype()
            # Visualize().update(cppn.createPhenotype())
            Visualize().update(phenotype)
            testOrganism(phenotype, displayEnv)
            highestFitness = fitness

        updated: bool = hyperneat.updateCandidate(cppn, fitness, behavior)
        if updated:
            total = pow(hyperneat.neat.population.configuration.mapResolution, len(hyperneat.neat.population.configuration.features))
            archiveFilled = len(hyperneat.neat.population.archivedGenomes)/total

            table = PrettyTable(["ID", "fitness", "max fitness", "distance", "max distance", "neurons", "links", "avg.weight", "archive"])
            table.add_row([
                cppn.ID,
                "{:1.4f}".format(fitness),
                "{:1.4f}".format(highestFitness),
                "{:1.4f}".format(distance),
                "{:1.4f}".format(highestDistance),
                len(cppn.neurons),
                len(cppn.links),
                "{:1.4f}".format(np.mean([l.weight for l in cppn.links])),
                "{:1.8f}".format(archiveFilled)])
            print(table)

    env.close()