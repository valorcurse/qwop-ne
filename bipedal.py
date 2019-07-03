from typing import List, Set, Dict, Tuple, Optional, Any

from genes import innovations
from neat import NEAT, CNeuralNet

from visualize import Visualize

from prettytable import PrettyTable
import numpy as np
import scipy as sp
import math

import os
import sys
import _pickle as pickle
import argparse

from platypus import NSGAII, Problem, Real

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

def testOrganism(env: Any, phenotype: CNeuralNet, novelty_map: Any, render: bool) -> Dict[str, Any]:
    observation = env.reset()
    
    rewardSoFar: float = 0.0
    distanceSoFar: float = 0.0
    
    actionsDone = np.zeros(8)
    behavior = np.zeros(14)

    sparseness: float = 0
    nrOfSteps: int = 0
    totalSpeed: float = 0

    previousDistance: float = 0.0
    rewardStagnation: int = 0 

    # if (render):
        # phenotype.draw()
        # Visualize(phenotype).draw()

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

        observation, reward, done, info = env.step(action)
        
        behavior += observation[:14]
        
        if (render):
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
            

        if done or rewardStagnation >= 100:
            # complete = True
            break

        rewardSoFar += reward
        previousDistance = distanceSoFar
        nrOfSteps += 1

    # actionsDone = np.divide(actionsDone, nrOfSteps)

    return {
        # "behavior": [actionsDone],
        "behavior": [behavior],
        "speed": totalSpeed,
        "nrOfSteps": nrOfSteps,
        "distanceTraveled": distanceSoFar
    }

class Bipedal(Problem):

    def __init__(self) -> None:
        super(Bipedal, self).__init__(4, 2)
        self.types = Real(-1000, 1000)


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

    neat = None
    novelty_map = None
    if (args.load):
        print("Loading NEAT object from file")
        with open("saves/" + args.load, "rb") as load:
            neat, innovations, novelty_map = pickle.load(load)
            neat.populationSize = 150
            # neat.milestone = 157.0
    else:
        print("Creating NEAT object")
        neat = NEAT(500, inputs, outputs, fullyConnected = False)
        novelty_map = np.empty((0, 14), float)

    # neat = None
    # print("Loading NEAT object from file")
    # with open("saves/lotsofspecies.ne", "rb") as load:
    # with open("saves/bipedal/bipedal.178", "rb") as load:
        # neat, innovations, novelty_map = pickle.load(load)

    # General settings
    IDToRender = None
    highestReward = [-1000.0, 0]
    # milestone: float = 1.0
    # nrOfSteps: int  = 600

    vis = Visualize()


    problem = Problem(4, 2)
    problem.types[:] = Real(-1000, 1000)
    problem.function = testOrganism

    while True:
        rewards: List[float] = []
        novelties: List[float] = []
        milestones: List[float] = []

        IDToRender = deepcopy(highestReward[1])
        highestReward = [-1000.0, 0]
        
        for i, phenotype in enumerate(neat.phenotypes):

            render = phenotype.ID == IDToRender

            if render:
                vis.update(phenotype)

            output = testOrganism(env, phenotype, novelty_map, render)

            distanceTraveled = output["distanceTraveled"]
            speed = output["speed"]/output["nrOfSteps"]
                # output = np.round(np.divide(output, nrOfSteps), decimals=4)

            sparseness = 0.0
            if novelty_map.size > 0:
                kdtree = sp.spatial.cKDTree(novelty_map)

                neighbours = kdtree.query(output["behavior"], k)[0]
                neighbours = neighbours[neighbours < 1E308]

                sparseness = (1/k)*np.sum(neighbours)


            if (novelty_map.size < k or sparseness > p_threshold):
                novelty_map = np.append(novelty_map, output["behavior"], axis=0)

            # np.sqrt(np.power(output["nrOfSteps"]*2, 2)*dimensions)
            # totalReward = output["distanceTraveled"] + output["distanceTraveled"]*(sparseness/output["nrOfSteps"])
            
            totalReward = sparseness + sparseness*(speed/neat.milestone)
            # totalReward = sparseness

            if (speed > neat.milestone):
                neat.milestone = speed

            if (totalReward > highestReward[0]):
                print("")
                print("Milestone: " + str(np.round(neat.milestone, 2)) + 
                    " | Distance Traveled: " + str(np.round(distanceTraveled, 2)) + 
                    " | Speed: " + str(np.round(speed, 2)) + 
                    " | Sparseness: " + str(np.round(sparseness, 2)) + 
                    # " | Actions done: " +str(np.round(output["actionsDone"], 2)) +
                    " | Total reward: " + str(np.round(totalReward, 2))
                )
                highestReward = [totalReward, phenotype.ID]

            rewards.append(totalReward)
            novelties.append(sparseness)
            milestones.append(output["distanceTraveled"])

            print("\rFinished phenotype ("+ str(i) +"/"+ str(len(neat.phenotypes)) +")", end='')

        print("\n")
        print("-----------------------------------------------------")
        # print(rewards)
        print("Running epoch")
        print("Generation: " + str(neat.generation))
        print("Number of species: " + str(len(neat.species)))
        print("Phase: " + str(neat.phase))
        print("Novelty map: " + str(novelty_map.data.shape))
        # print("Milestones: " + str(milestones))
        table = PrettyTable(["ID", "age", "members", "min. milestone", "max milestone", "max fitness", "adj. fitness", "max distance", "stag", "neurons", "links", "avg.weight", "avg. bias", "avg. compat.", "to spawn"])
        for s in neat.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                s.milestone,
                round(max([m.milestone for m in s.members]), 4),                   # Max milestone
                round(max([m.fitness for m in s.members]), 4),                   # Max novelty
                round(max([m.adjustedFitness for m in s.members]), 4),                   # Max fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
                s.generationsWithoutImprovement,                            # Stagnation
                "{:1.4f}".format(np.mean([len(m.neurons)-28 for m in s.members])),     # Neurons
                "{:1.4f}".format(np.mean([len(m.links) for m in s.members])),                 # Links
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        for index, genome in enumerate(neat.genomes):
            genome.milestone = milestones[index]

        neat.epoch(rewards)

        # Save current generation
        saveFileName = saveFolder + "." + str(neat.generation)
        with open(saveDirectory + "/" + saveFileName, 'wb') as binaryFile:
            pickle.dump([neat, innovations, novelty_map], binaryFile)

        # Append to summary file
        with open(saveDirectory + "/summary.txt", 'a') as textFile:
            textFile.write("Generation " + str(neat.generation) + "\n")
            textFile.write(table.get_string())
            textFile.write("\n\n")

    env.close()