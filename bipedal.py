from typing import List, Set, Dict, Tuple, Optional, Any

from neat.hyperneat import HyperNEAT
from neat.genes import Genome
from neat.phenotypes import Phenotype
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
    env = displayEnv if displayEnv is not None else gym.make('BipedalWalker-v2')
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


    # if (displayEnv):
        # phenotype.draw()
        # Visualize().update(phenotype)

    # while not complete:
    for _ in range(max_frames):
# 
        action = phenotype.update(observation)


        # firstAction = [action[0], 0.0] if action[0] > 0 else [0.0, action[0]]
        # secondAction = [action[1], 0.0] if action[1] > 0 else [0.0, action[1]]
        # thirdAction = [action[2], 0.0] if action[2] > 0 else [0.0, action[2]]
        # fourthAction = [action[3], 0.0] if action[3] > 0 else [0.0, action[3]]

        # splitActions = [item for sublist in [firstAction, secondAction, thirdAction, fourthAction] for item in sublist]
        # splitActions = [abs(number) for number in splitActions]

        # actionsDone += np.array(np.absolute(splitActions))
        # links = [n.linksIn for n in phenotype.neurons]
        # links = [item for sublist in links for item in sublist]
        # if len(links) > 0:
        #     print("\r" + str(action) + " " + str(len(links)), end='')
        
        observation, reward, done, info = env.step(action)
        
        # behavior += observation[:14]
        behavior += [
            observation[0], 
            observation[2], 
            # observation[4],
            # observation[6],
            observation[8],
            # observation[9],
            # observation[11]
            observation[13]
        ]
        
        if (displayEnv is not None):
            env.render()
            # print(action)
        
        speed: float = round(observation[2], 1)
        totalSpeed += speed

        distanceSoFar += speed
        distanceDiff: float = distanceSoFar-previousDistance
        # print("\r", rewardStagnation, "\t", distanceDiff, end='')

        if distanceDiff > 0.0:
            rewardStagnation = 0
        else:
            rewardStagnation += 1
            
        fitness = distanceSoFar
        if done or rewardStagnation >= 100:
            if done:
                fitness *= 0.5

            break

        rewardSoFar += round(reward, 1)
        previousDistance = distanceSoFar
        nrOfSteps += 1

    # actionsDone = np.divide(actionsDone, nrOfSteps)
    behavior = np.divide(behavior, nrOfSteps)

    # print(behavior)

    return {
        # "behavior": [actionsDone],
        "behavior": behavior,
        # "speed": totalSpeed,
        # "nrOfSteps": nrOfSteps,
        "distanceTraveled": distanceSoFar,
        # "fitness": rewardSoFar
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

    env = gym.make('BipedalWalker-v2')

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

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

    randomPop = hyperneat.neat.population.randomInitialization()
    for i, genome in enumerate(randomPop):
        print("\rInitializing start population ("+ str(i) +"/"+ str(len(randomPop)) +")", end='')
        output = testOrganism(hyperneat.createSubstrate(genome).createPhenotype())
        hyperneat.epoch(genome, output["fitness"], output["behavior"])

    displayEnv = gym.make('BipedalWalker-v2')
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
        output = pool.map(testOrganism, [candidate.createPhenotype()]*5)
        fitness = max([l["fitness"] for l in output])
        distance = max([l["distanceTraveled"] for l in output])
        behavior = np.array([l["behavior"] for l in output])
        behavior = np.add.reduce(behavior)/runs
        # for _ in range(runs):
        #     phenotype = hyperneat.createSubstrate(cppn).createPhenotype()
        #     Visualize().update(phenotype)
        #     output = testOrganism(phenotype, displayEnv)
        
        #     fitness = max(fitness, output["fitness"])
        #     distance = max(distance, output["distanceTraveled"])
        #     behavior += output["behavior"]

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

    # while True:
    #     rewards: List[float] = []
    #     novelties: List[float] = []
    #     milestones: List[float] = []

    #     IDToRender = deepcopy(highestReward[1])
    #     highestReward = [-1000.0, 0]
        
        # for i, phenotype in enumerate(hyperneat.phenotypes):

        #     render = phenotype.ID == IDToRender

        #     # if render:
        #         # vis.update(phenotype)

        #     output = testOrganism(env, phenotype, novelty_map, render)

        #     distanceTraveled = output["distanceTraveled"]
        #     speed = output["speed"]/output["nrOfSteps"]
        #         # output = np.round(np.divide(output, nrOfSteps), decimals=4)

        #     sparseness = 0.0
        #     if novelty_map.size > 0:
        #         kdtree = sp.spatial.cKDTree(novelty_map)

        #         neighbours = kdtree.query(output["behavior"], k)[0]
        #         neighbours = neighbours[neighbours < 1E308]

        #         sparseness = (1/k)*np.sum(neighbours)


        #     if (novelty_map.size < k or sparseness > p_threshold):
        #         novelty_map = np.append(novelty_map, output["behavior"], axis=0)

        #     # np.sqrt(np.power(output["nrOfSteps"]*2, 2)*dimensions)
        #     # totalReward = output["distanceTraveled"] + output["distanceTraveled"]*(sparseness/output["nrOfSteps"])
            
        #     totalReward = sparseness + sparseness*(speed/hyperneat.milestone)
        #     # totalReward = sparseness

        #     if (speed > hyperneat.milestone):
        #         hyperneat.milestone = speed

        #     if (totalReward > highestReward[0]):
        #         print("")
        #         print("Milestone: " + str(np.round(hyperneat.milestone, 2)) + 
        #             " | Distance Traveled: " + str(np.round(distanceTraveled, 2)) + 
        #             " | Speed: " + str(np.round(speed, 2)) + 
        #             " | Sparseness: " + str(np.round(sparseness, 2)) + 
        #             # " | Actions done: " +str(np.round(output["actionsDone"], 2)) +
        #             " | Total reward: " + str(np.round(totalReward, 2))
        #         )
        #         highestReward = [totalReward, phenotype.ID]

        #     rewards.append(totalReward)
        #     novelties.append(sparseness)
        #     milestones.append(output["distanceTraveled"])

        #     print("\rFinished phenotype ("+ str(i) +"/"+ str(len(hyperneat.phenotypes)) +")", end='')

        # print("\n")
        # print("-----------------------------------------------------")
        # # print(rewards)
        # print("Running epoch")
        # print("Generation: " + str(hyperneat.generation))
        # print("Number of species: " + str(len(hyperneat.population.species)))
        # print("Phase: " + str(hyperneat.phase))
        # print("Novelty map: " + str(novelty_map.data.shape))
        # # print("Milestones: " + str(milestones))
        # table = PrettyTable(["ID", "age", "members", "min. milestone", "max milestone", "max fitness", "adj. fitness", "max distance", "stag", "neurons", "links", "avg.weight", "avg. bias", "avg. compat.", "to spawn"])
        # for s in hyperneat.population.species:
        #     table.add_row([
        #         s.ID,                                                       # Species ID
        #         s.age,                                                      # Age
        #         len(s.members),                                             # Nr. of members
        #         s.milestone,
        #         round(max([m.milestone for m in s.members]), 4),                   # Max milestone
        #         round(max([m.fitness for m in s.members]), 4),                   # Max novelty
        #         round(max([m.adjustedFitness for m in s.members]), 4),                   # Max fitness
        #         "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
        #         s.generationsWithoutImprovement,                            # Stagnation
        #         "{:1.4f}".format(np.mean([len(m.neurons)-28 for m in s.members])),     # Neurons
        #         "{:1.4f}".format(np.mean([len(m.links) for m in s.members])),                 # Links
        #         "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
        #         "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
        #         "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
        #         s.numToSpawn])                                              # Nr. of members to spawn

        # print(table)

        # for index, genome in enumerate(hyperneat.population.genomes):
        #     genome.milestone = milestones[index]

        # hyperneat.epoch(rewards)

        # # Save current generation
        # saveFileName = saveFolder + "." + str(hyperneat.generation)
        # with open(saveDirectory + "/" + saveFileName, 'wb') as binaryFile:
        #     pickle.dump([hyperneat, innovations, novelty_map], binaryFile)

        # # Append to summary file
        # with open(saveDirectory + "/summary.txt", 'a') as textFile:
        #     textFile.write("Generation " + str(hyperneat.generation) + "\n")
        #     textFile.write(table.get_string())
        #     textFile.write("\n\n")

    env.close()