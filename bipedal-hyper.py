import argparse
import os
import sys
import threading
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from typing import List, Dict, Any

import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import numpy as np
from prettytable import PrettyTable

from numba import vectorize, jit

import neat.hyperneat as hn
from neat.neatTypes import NeuronType
from neat.phenotypes import Phenotype, FeedforwardCUDA
from neat.speciatedPopulation import SpeciesConfiguration, SpeciesUpdate

import time

env_name = "BipedalWalker-v2"
nproc = 4

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

# _ALE_LOCK = threading.Lock()

def take_step(env_action):
    env, action = env_action

    state, reward, done, info = env.step(action)
    # with _ALE_LOCK:
        # env.render()

    return np.array([state, reward, done])

def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

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

    env = gym.make(env_name)

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
    pool = Pool(4)

    highestFitness: float = 0.0
    highestDistance: float = 0.0

    while True:

        fitness = 0.0
        distance = 0.0


        feedforward = FeedforwardCUDA(phenotypes)

        envs = [make_env(env_name, seed) for seed in range(len(phenotypes))]
        envs = SubprocVecEnv(envs)
        # envs = DummyVecEnv(envs)

        observations = envs.reset()
        # for e in envs:
        #     observations.append(e.reset())

        # print(observations.shape)

        obs_32 = np.float32(observations)
        actions = feedforward.update(obs_32)

        fitnesses = np.zeros(len(phenotypes), dtype=np.float64)

        done = False

        done_tracker = np.array([False for _ in phenotypes])

        start = time.time()
        while not done:
            # round_start = time.time()

            states, rewards, dones, info = envs.step(actions)
            actions = feedforward.update(states)


            rewards = rewards.astype(np.float64)
            fitnesses[dones == False] += rewards[dones == False]


            envs_done = dones == True
            done_tracker[envs_done] = dones[envs_done]
            envs_running = len([d for d in done_tracker if d == False])
            done = envs_running == 0

            # round_end = time.time()
            # print("Round time: {} | Envs running: {}/{}".format(round_end - round_start, envs_running, len(phenotypes)))

        end = time.time()
        print("Time:", end - start)

        print("Fitnesses:", fitnesses)
        max_fitness = max(fitnesses)

        if max_fitness > highestFitness:
            print("New highest fitness: %f"%(max_fitness))
            highestFitness = max_fitness

        print("Highest fitness: {}".format(max_fitness))

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
