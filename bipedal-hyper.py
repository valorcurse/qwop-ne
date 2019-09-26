import argparse
import os
import sys
import threading
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Dict, Any

import gym
import numpy as np
from prettytable import PrettyTable

from numba import vectorize, jit

import neat.hyperneat as hn
from neat.phenotypes import Phenotype, FeedforwardCUDA
from neat.speciatedPopulation import SpeciesConfiguration, SpeciesUpdate
from neat.neatTypes import NeuronType

import time

environment = "BipedalWalker-v2"

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

_ALE_LOCK = threading.Lock()

def take_step(env_action):
    env, action = env_action

    state, reward, done, info = env.step(action)
    # with _ALE_LOCK:
        # env.render()

    return np.array([state, reward, done])

# @jit
# def take_steps(envs, actions):
#     # print(envs)
#     observations = []
#     for action, env in zip(actions, envs):
#         observation = env.step(action)
#         # env.render()
#         # print(observation)
#         observations.append(observation)
#
#     # print("Observations: {}".format(np.array(observations)))
#     return np.array(observations).T


# def testOrganisms(feedforward, envs):
#     X = np.array([e.reset() for e in envs])
#
#     all_rewards = np.zeros(X.shape[0])
#
#     done = False
#     while not done:
#         actions = feedforward.update(X)
#         X, rewards, dones, info = take_steps(envs, actions)
#         # print(all_rewards.shape, rewards.shape)
#         all_rewards = np.add(all_rewards, rewards)
#         done = len([d for d in dones if d == False]) == 0
#         # print(done, all_rewards)
#
#     return all_rewards


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

    # inputs = env.observation_space.sample().size
    # # outputs = env.action_space[0]
    # outputs = 18
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))
    print("Creating hyperneat object")
    pop_size = 100
    pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    hyperneat = hn.HyperNEAT(pop_config)

    # fitnesses: List[float] = []
    start_fitness = [0.0]*pop_size
    phenotypes: List[Phenotype] = hyperneat.epoch(SpeciesUpdate(start_fitness))
    # print("Phenotypes: {}".format(phenotypes))
    # randomPop = hyperneat.neat.population.randomInitialization()
    # for i, genome in enumerate(randomPop):
    #     print("\rInitializing start population ("+ str(i) +"/"+ str(len(randomPop)) +")", end='')
    #     fitnesses.append(testOrganism(hyperneat.createSubstrate(genome).createPhenotype())["fitness"])
    # phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))

    # displayEnv = gym.make(environment)
    # displayEnv.render()
    pool = ThreadPool(4)

    highestFitness: float = 0.0
    highestDistance: float = 0.0

    while True:

        fitness = 0.0
        distance = 0.0


        feedforward = FeedforwardCUDA(phenotypes)
        # print("Feedforward: {}".format(feedforward))
        envs = [gym.make(environment) for _ in phenotypes]
        # print("Envs: {}".format(envs))
        # observations = map(lambda e: e.reset(), envs)
        observations = []
        for e in envs:
            observations.append(e.reset())
        # print("Observation: {}".format(observations))

        actions = feedforward.update(observations)
        print(actions)

        fitnesses = np.zeros(len(phenotypes), dtype=np.float64)

        done = False

        start = time.time()
        while not done:
            # Parallel
            parallel_envs = list(zip(envs, actions))
            outputs = np.array(pool.map(take_step, parallel_envs))

            states, rewards, dones = outputs.T

            # feed_start = time.time()
            actions = feedforward.update(states)
            print(actions)
            # feed_end = time.time()
            # print("Feedfoward Time:", feed_end - feed_start)

            envs_running = len([d for d in dones if d == False])
            done = envs_running == 0

            rewards = rewards.astype(np.float64)
            fitnesses[dones == False] += rewards[dones == False]

        end = time.time()
        print("Time:", end - start)

        print("Fitnesses:", fitnesses)
        max_fitness = max(fitnesses)

        # for phenotype in phenotypes:
        #     print("Phenotype: %d | Neurons: %d | Links: %d"%
        #         (phenotype.ID, len(phenotype.graph.nodes), len(phenotype.graph.edges))
        #         )
            # Visualize().update(phenotype)
        #     output = testOrganism(phenotype, None)
        #     fitness = max(fitness, output["fitness"])
        #     fitnesses.append(fitness)

        if max_fitness > highestFitness:
            print("New highest fitness: %f"%(max_fitness))
            # done = False
            # states = env.reset()

            # while not done:
            #     actions = feedforward.update(states)
            #     states, rewards, dones = take_step(list(zip(env, actions)))
            #     env.render()
            # Visualize().update(cppn.createPhenotype())
            # Visualize().update(phenotype)
            # testOrganism(phenotype, displayEnv)
            highestFitness = fitness

        table = PrettyTable(["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. compat.", "to spawn"])
        for s in hyperneat.neat.population.species:
            table.add_row([
                s.ID,                                                       # Species ID
                s.age,                                                      # Age
                len(s.members),                                             # Nr. of members
                int(max([m.fitness for m in s.members])),                   # Max fitness
                # "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
                "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
                # "{}".format(max([m.uniqueKeysPressed for m in s.members])), # Average unique keys
                s.generationsWithoutImprovement,                            # Stagnation
                int(np.mean([len([n for n in m.neurons if n.neuronType == NeuronType.HIDDEN]) for m in s.members])),     # Neurons
                np.mean([len(m.links) for m in s.members]),                 # Links
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
                # "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
                s.numToSpawn])                                              # Nr. of members to spawn

        print(table)

        phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))

    env.close()
