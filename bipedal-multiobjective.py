from typing import List, Dict, Any, Tuple

import argparse
import os
import sys
import psutil

import multiprocessing as mp

import gym
import numpy as np
import pygmo as pg
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from visualize import Visualize
from prettytable import PrettyTable

# import neat.hyperneat as hn
from neat.neat import NEAT, Evaluation
from neat.neatTypes import NeuronType
from neat.noveltySearch import NoveltySearch
from neat.phenotypes import Phenotype, FeedforwardCUDA
from neat.MultiObjectivePopulation import MOConfiguration, MOUpdate

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

env_name = "BipedalWalker-v2"
env = gym.make(env_name)
nproc = 4

envs_size = 10
pop_size = 10
max_stagnation = 25
encoding_dim = 8
behavior_dimensions = 7
behavior_steps = 60
behavior_matrix_size = behavior_dimensions * behavior_steps

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

class TestOrganism(Evaluation):

    def __init__(self):
        print("Creating envs...")
        self.envs = SubprocVecEnv([make_env(env_name, seed) for seed in range(envs_size)])
        self.num_of_envs = envs_size
        print("Done.")


    def evaluate(self, phenotypes) -> Tuple[np.ndarray, np.ndarray]:
        feedforward = FeedforwardCUDA(phenotypes)

        observations = self.envs.reset()

        obs_32 = np.float32(observations)
        actions = feedforward.update(obs_32)

        fitnesses = np.zeros(len(self.envs.remotes), dtype=np.float64)

        done = False
        done_tracker = np.array([False for _ in range(len(self.envs.remotes))])

        diff = len(phenotypes) - len(self.envs.remotes)
        if diff < 0:
            done_tracker[diff:] = True

        distances = np.zeros(len(self.envs.remotes))
        last_distances = np.zeros(len(self.envs.remotes))
        stagnations = np.zeros(len(self.envs.remotes))

        all_states = []


        max_steps = 50
        steps = max_steps
        while not done:
            actions = np.pad(actions, (0, abs(diff)), 'constant')
            states, rewards, dones, info = self.envs.step(actions)

            # if render:
            #     envs.remotes[0].send(('render', None))
            #     envs.remotes[0].recv()

            actions = feedforward.update(states)

            fitnesses[done_tracker == False] += np.around(rewards[done_tracker == False], decimals=2)
            # fitnesses[done_tracker == False] = np.around(rewards[done_tracker == False], decimals=2)

            envs_done = dones == True
            done_tracker[envs_done] = dones[envs_done]
            envs_running = len([d for d in done_tracker if d == False])

            # print("\r"+" "* 100, end='', flush=True)
            # print("\rEnvs running: {}/{}".format(envs_running, len(phenotypes)), end='')


            done = envs_running == 0

            distances += np.around(states.T[2], decimals=2)

            stagnations += distances == last_distances

            done_tracker[stagnations >= 100] = True

            last_distances = distances

            if steps == max_steps:
                steps = 0
                all_states.append(states[:, [0, 4, 6, 8, 9, 11, 13]])

            steps += 1

        all_states = np.array(all_states)
        flattened_states = []
        for row_i in range(all_states.shape[1]):
            flattened_states.append(all_states[:, row_i].flatten())

        flattened_states = pad_matrix(np.array(flattened_states), behavior_matrix_size)

        return (fitnesses, flattened_states)

# def chunk_testing(phenotypes):
#     all_states = []
#     fitnesses = []
#     for chunk in chunks(phenotypes, envs_size):
#         fitness, states = test_organism(chunk, envs)
#         all_states.extend(states)
#         fitnesses.extend(fitness)
#
#     return (np.array(all_states), np.array(fitnesses))

def run_env_once(phenotype):
    # single_envs = SubprocVecEnv([make_env(env_name, envs_size)])
    # test_organism([phenotype], single_envs, render=True)

    # Visualize().update(phenotype)
    feedforward_highest = FeedforwardCUDA([phenotype])
    states = env.reset()
    done = False
    distance = 0.0
    last_distance = 0.0
    distance_stagnation = 0
    while not done:
        actions = feedforward_highest.update(np.array([states]))
        states, reward, done, info = env.step(actions[0])
        distance += np.around(states[2], decimals=2)

        if distance <= last_distance:
            distance_stagnation += 1
        else:
            distance_stagnation = 0

        if distance_stagnation >= 100:
            done = True

        last_distance = distance

        env.render()
    # env.close()
    # Visualize().close()

# def refine_ae(autoencoder, phenotypes):
#     autoencoder.load_weights('basic.h5')
# 
#     last_loss = 1000.0
#     loss = 1000.0
# 
#     while loss <= last_loss:
#         last_loss = loss
# 
#         states, _ = chunk_testing(phenotypes)
# 
#         ten_percent = max(1, int(states.shape[0] * 0.1))
# 
#         train = states[:-ten_percent]
#         test = states[-ten_percent:]
# 
#         hist = autoencoder.fit(train, train,
#                                epochs=50,
#                                batch_size=256,
#                                shuffle=True,
#                                validation_data=(test, test),
#                                verbose=0)
# 
#         loss = abs(hist.history['val_loss'][-1])
# 
#         print("Training autoencoder. Loss: {}".format(loss))

def pad_matrix(all_states, matrix_width):
    padded = []
    for row in all_states:
        # row = all_states[:, i].flatten()

        row = np.pad(row, (0, abs(matrix_width - row.shape[0])), 'constant')
        padded.append(row)

    return np.array(padded)

# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

if __name__ == '__main__':

    # import keras
    # from keras.layers import Input, Dense
    # from keras.models import Model
    # from keras.backend.tensorflow_backend import set_session
    # import tensorflow as tf
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    sys.setrecursionlimit(10000)

    # env = gym.make(env_name)

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    # novelty_map = NoveltySearch(encoding_dim)

    ############################################## Auto encoder ##############################################

    # # this is our input placeholder
    # input_img = Input(shape=(behavior_matrix_size,))
    # # "encoded" is the encoded representation of the input
    # encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(behavior_matrix_size, activation='relu')(encoded)
    #
    # # this model maps an input to its reconstruction
    # autoencoder = Model(input_img, decoded)
    #
    # # this model maps an input to its encoded representation
    # encoder = Model(input_img, encoded)
    #
    # # create a placeholder for an encoded (32-dimensional) input
    # encoded_input = Input(shape=(encoding_dim,))
    # # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(encoded_input))
    #
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # autoencoder.save_weights('basic.h5')
    ############################################################################################################

    print("Creating hyperneat object")
    # pop_config = MapElitesConfiguration(4, pop_size, encoding_dim, inputs, outputs)
    # hyperneat = hn.HyperNEAT(pop_config)
    pop_config = MOConfiguration(pop_size, inputs, outputs)
    neat = NEAT(TestOrganism(), pop_config)

    start_fitness = np.zeros(pop_size)
    start_features = np.zeros((pop_size, encoding_dim))
    # phenotypes: List[Phenotype] = hyperneat.epoch(MapElitesUpdate(start_fitness, start_features))
    # phenotypes: List[Phenotype] = neat.epoch(SpeciesUpdate(start_fitness))

    # highest_fitness: float = -1000.0
    # highest_novelty: float = 0.0
    # highestDistance: float = 0.0


    # envs = [make_env(env_name, seed) for seed in range(envs_size)]
    #
    # print("Creating envs...")
    # envs = SubprocVecEnv(envs)
    # print("Done.")

    # loss = 1000.0
    # progress_stagnation = 0
    # train_ae = progress_stagnation == max_stagnation

    # refine_ae(autoencoder, phenotypes)


    # epoch_num = 0
    # while True:
    #     epoch_num += 1
    #     progress_stagnation += 1
    #     train_ae = progress_stagnation == max_stagnation
    #
    #     if train_ae:
    #         ae_phenotypes = []
    #         if len(neat.population) > 0:
    #
    #             sorted_archive = sorted(neat.population.population_and_fitnesses(), key=lambda a: a['fitness'])
    #             ae_phenotypes = [g['genome'].createPhenotype() for g in sorted_archive]
    #         else:
    #             ae_phenotypes = phenotypes
    #
    #         refine_ae(autoencoder, ae_phenotypes)
    #         ten_percent = max(1, int(len(ae_phenotypes) * 0.1))
    #         # phenotypes = ae_phenotypes[:ten_percent]
    #
    #         # Re-evaluate all the genomes in the archives
    #         ae_states, _ = chunk_testing(ae_phenotypes[:ten_percent])
    #         ae_pred = encoder.predict(ae_states)
    #
    #         novelty_map.reset()
    #         novelty_map.calculate_novelty(ae_pred)
    #
    #         neat.population.archive = {}
    #         progress_stagnation = 0



        # Test the phenotypes in the envs
        # start = time.time()
        # all_states, fitnesses = chunk_testing(phenotypes)
        # end = time.time()
        # print("Time:", end - start)
        #
        #
        # pred = encoder.predict(all_states)
        # novelties = novelty_map.calculate_novelty(pred)

        # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=zip(fitnesses, novelties))

        # print("Highest fitness this epoch:", max(fitnesses))


    for _ in neat.epoch():

        # max_fitness = max(zip(fitnesses, phenotypes), key=lambda e: e[0])
        # max_novelty = max(zip(novelties, phenotypes), key=lambda e: e[0])
        #
        # if max_fitness[0] > highest_fitness or max_novelty[0] > highest_novelty:
        #     highest_fitness = max_fitness[0] if max_fitness[0] > highest_fitness else highest_fitness
        #     highest_novelty = max_fitness[0] if max_fitness[0] > highest_fitness else highest_novelty
        #
        #     best_phenotype = max_fitness[1]
        #     run_env_once(best_phenotype)

        print("########## Epoch {} ##########".format(neat.epochs))
        # print("Highest fitness all-time: {}".format(highest_fitness))
        # print("Progress stagnation: {}".format(progress_stagnation))

        table = PrettyTable(["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. compat.", "to spawn"])
        for s in neat.population.species:
            table.add_row([
                # Species ID
                s.ID,
                # Age
                s.age,
                # Nr. of members
                len(s.members),
                # Max fitness
                "{:1.4f}".format(max([m.fitness for m in s.members])),
                # Average distance
                "{:1.4f}".format(max([m.distance for m in s.members])),
                # Stagnation
                s.generationsWithoutImprovement,
                # Neurons
                np.mean([len([n for n in m.neurons if n.neuronType == NeuronType.HIDDEN]) for m in s.members]),
                # Links
                np.mean([len(m.links) for m in s.members]),
                # Avg. weight
                "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),
                # Avg. compatiblity
                "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),
                # Nr. of members to spawn
                s.numToSpawn])

        print(table)

        # phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))
        # phenotypes = hyperneat.epoch(SpeciesUpdate(novelties))
        # phenotypes = hyperneat.epoch(MapElitesUpdate(fitnesses, pred))

    env.close()
