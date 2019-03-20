from genes import innovations
from neat import NEAT
from prettytable import PrettyTable
import numpy as np
import scipy as sp

import os
import sys
import _pickle as pickle

from copy import deepcopy

import gym

k: int = 15
p_threshold: float = 3.0

def testOrganism(env, phenotype, novelty_map, render):
    observation = env.reset()
    complete = False
    
    totalReward: float = 0.0
    rewardSoFar: float = 0.0
    distanceSoFar: float = 0.0
    
    actionsDone = np.zeros(4)
    nrOfSteps = 0

    sparseness: float = 0
    nrOfSteps: int = 0
    totalSpeed: float = 0

    previousDistance: float = 0.0
    rewardStagnation: int = 0 

    while not complete:
        if (render):
            env.render()

        complete = False

        action = phenotype.update(observation)

        actionsDone += np.array(np.absolute(action))

        observation, reward, done, info = env.step(action)
        
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
            complete = True

        rewardSoFar += reward
        previousDistance = distanceSoFar
        nrOfSteps += 1

    return [actionsDone]



# gym.envs.register(
#     id='CartPole-v2',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     max_episode_steps=500000
# )
if __name__ == '__main__':

    env = gym.make('BipedalWalker-v2')

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    neat = NEAT(50, inputs, outputs)

    # General settings
    IDToRender = None
    highestReward = [0, 0]
    nrOfSteps: int  = 600

    # Novelty search
    nrOfDimensions = nrOfSteps + 4
    novelty_map = np.empty((0, 4), float)

    # Save generations to file
    saveFolder = "bipedal"
    saveDirectory = "saves/" + saveFolder
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    while True:
        rewards: float = []
        novelties: float = []

        IDToRender = deepcopy(highestReward[1])
        highestReward = [0, 0]
        
        for i, phenotype in enumerate(neat.phenotypes):

            print(novelty_map.size)

            actions_done = testOrganism(env, phenotype, novelty_map, phenotype.ID == IDToRender)

                # actionsDone = np.round(np.divide(actionsDone, nrOfSteps), decimals=4)

            if novelty_map.size > 0:
                kdtree = sp.spatial.KDTree(novelty_map)

                neighbours = kdtree.query(actionsDone, k)[0]
                neighbours = neighbours[neighbours < 1E308]

                sparseness = (1/k)*np.sum(neighbours)


            # sparseness = sparseness * (totalSpeed/nrOfSteps)

            if (novelty_map.size < k or sparseness > p_threshold):
                novelty_map = np.append(novelty_map, actions_done, axis=0)

            # rewardSoFar += rewardSoFar*sparseness
            # totalReward = 0.2*rewardSoFar + 0.8*sparseness

            # totalReward = max(rewardSoFar, sparseness) * (totalSpeed/nrOfSteps)
            totalReward = sparseness * (totalSpeed/nrOfSteps)

            # print(sparseness)

            if (reward[0] > highestReward[0]):
                highestReward = [reward[0], phenotype.ID]

            rewards.append(reward[0])
            novelties.append(reward[1])

            print("\rFinished phenotype ("+ str(i) +"/"+ str(len(neat.phenotypes)) +")", end='')

        print("\n")
        print("-----------------------------------------------------")
        # print(rewards)
        print("Running epoch")
        print("Generation: " + str(neat.generation))
        print("Number of species: " + str(len(neat.species)))
        print("Phase: " + str(neat.phase))
        print("Novelty map: " + str(novelty_map.data.shape))
        table = PrettyTable(["ID", "age", "members", "novelty", "max fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. bias", "avg. compat.", "to spawn"])
        for s in neat.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                round(max([m.novelty for m in s.members]), 4),                   # Max novelty
                round(max([m.fitness for m in s.members]), 4),                   # Max fitness
                # "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
                # "{}".format(max([m.uniqueKeysPressed for m in s.members])), # Average unique keys
                s.generationsWithoutImprovement,                            # Stagnation
                "{:1.4f}".format(np.mean([len(m.neurons)-6 for m in s.members])),     # Neurons
                "{:1.4f}".format(np.mean([len(m.links) for m in s.members])),                 # Links
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        neat.epoch(rewards, novelties)

        # Save current generation
        saveFileName = saveFolder + "." + str(neat.generation)
        with open(saveDirectory + "/" + saveFileName, 'wb') as file:
            pickle.dump([neat, innovations], file)

        # Append to summary file
        with open(saveDirectory + "/summary.txt", 'a') as file:
            file.write("Generation " + str(neat.generation) + "\n")
            file.write(table.get_string())
            file.write("\n\n")

    env.close()