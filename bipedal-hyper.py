from typing import List, Dict, Any

import argparse
import os
import sys
import psutil

import gym
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from visualize import Visualize
from prettytable import PrettyTable

import neat.hyperneat as hn
from neat.neatTypes import NeuronType
from neat.phenotypes import Phenotype, FeedforwardCUDA
# from neat.speciatedPopulation import SpeciesConfiguration, SpeciesUpdate
from neat.mapElites import MapElitesConfiguration, MapElitesUpdate

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env_name = "BipedalWalker-v2"
nproc = 4

dimensions: int = 8
k: int = 15
max_frames: int = 1000
maxMapDistance: float = 300.0

space_range: int = 50
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

    import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

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

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    ############################################## Auto encoder ##############################################

    encoding_dim = 4
    # this is our input placeholder
    input_img = Input(shape=(400,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(400, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    ############################################################################################################

    print("Creating hyperneat object")
    pop_size = 100
    # pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    pop_config = MapElitesConfiguration(64, pop_size, encoding_dim, inputs, outputs)
    # hyperneat = hn.HyperNEAT(pop_config)
    hyperneat = hn.NEAT(pop_config)

    start_fitness = [0.0]*pop_size
    start_features = np.zeros((pop_size, encoding_dim))
    phenotypes: List[Phenotype] = hyperneat.epoch(MapElitesUpdate(start_fitness, start_features))

    highestFitness: float = -1000.0
    highestDistance: float = 0.0

    envs = [make_env(env_name, seed) for seed in range(len(phenotypes))]

    print("Creating envs...")
    envs = SubprocVecEnv(envs)
    print("Done.")

    loss = 1000.0
    train_ae = True

    epoch_num = 0
    while True:
        epoch_num += 1
        print("########## Epoch {} ##########".format(epoch_num))

        diff = len(phenotypes) - len(envs.remotes)
        if diff > 0:
            envs.add_envs(diff)

        envs.reset()

        feedforward = FeedforwardCUDA(phenotypes)

        observations = envs.reset()

        obs_32 = np.float32(observations)
        actions = feedforward.update(obs_32)

        fitnesses = np.zeros(len(phenotypes), dtype=np.float64)

        done = False
        done_tracker = np.array([False for _ in range(len(envs.remotes))])

        if diff < 0:
            done_tracker[diff:] = True

        distances = np.zeros(len(envs.remotes))
        last_distances = []
        stagnations = np.zeros(len(envs.remotes))


        all_states = []

        start = time.time()
        max_steps = 50
        steps = max_steps
        while not done:
            actions = np.pad(actions, (0, abs(diff)), 'constant')
            states, rewards, dones, info = envs.step(actions)
            actions = feedforward.update(states)

            fitnesses[done_tracker == False] += rewards[done_tracker == False]

            envs_done = dones == True
            done_tracker[envs_done] = dones[envs_done]
            envs_running = len([d for d in done_tracker if d == False])

            # print(" " * 100, end='\r', flush=True)
            print("Envs running: {}/{}".format(envs_running, len(phenotypes)), end='\r')

            done = envs_running == 0

            distances += states.T[2]

            stagnations += distances == last_distances

            done_tracker[stagnations >= 100] = True

            last_distances = distances

            if steps == max_steps:
                steps = 0
                all_states.append(states[:, [4, 6, 9, 11]])

            steps += 1



        behavior_matrix_size = 400
        all_states = np.array(all_states)
        padded = []
        for i in range(all_states.shape[1]):
            row = all_states[:, i].flatten()
            diff = behavior_matrix_size - len(row)

            row = np.pad(row, (0, abs(behavior_matrix_size - row.shape[0])), 'constant')
            padded.append(row)

        padded = np.array(padded)

        ten_percent = int(padded.shape[0] * 0.1)

        if train_ae:

            train = padded[:-ten_percent]
            test = padded[-ten_percent:]

            hist = autoencoder.fit(train, train,
                                   epochs=50,
                                   batch_size=256,
                                   shuffle=True,
                                   validation_data=(test, test),
                                   verbose=0)

            lastest_loss = abs(hist.history['val_loss'][-1])
            if lastest_loss > loss:
                train_ae = False

            loss = lastest_loss
            print("Training autoencoder. Loss: {}".format(lastest_loss))

            continue


        pred = encoder.predict(padded)

        end = time.time()
        print("Time:", end - start)

        print("Highest fitness this epoch:", max(fitnesses))
        max_fitness = max(zip(fitnesses, phenotypes), key=lambda e: e[0])

        if max_fitness[0] > highestFitness:
            print("New highest fitness: {}".format(max_fitness))
            best_phenotype = max_fitness[1]
            # Visualize().update(best_phenotype)

            feedforward_highest = FeedforwardCUDA([best_phenotype])
            states = env.reset()
            done = False
            while not done:
                actions = feedforward_highest.update(np.array([states]))
                states, reward, done, info = env.step(actions[0])
                env.render()

            # Visualize().close()
            highestFitness = max_fitness[0]

        print("Highest fitness all-time: {}".format(highestFitness))

        # table = PrettyTable(["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg.weight", "avg. compat.", "to spawn"])
        # for s in hyperneat.neat.population.species:
        #     table.add_row([
        #         s.ID,                                                       # Species ID
        #         s.age,                                                      # Age
        #         len(s.members),                                             # Nr. of members
        #         int(max([m.fitness for m in s.members])),                   # Max fitness
        #         # "{:1.4f}".format(s.adjustedFitness),                        # Adjusted fitness
        #         "{:1.4f}".format(max([m.distance for m in s.members])),          # Average distance
        #         # "{}".format(max([m.uniqueKeysPressed for m in s.members])), # Average unique keys
        #         s.generationsWithoutImprovement,                            # Stagnation
        #         int(np.mean([len([n for n in m.neurons if n.neuronType == NeuronType.HIDDEN]) for m in s.members])),     # Neurons
        #         np.mean([len(m.links) for m in s.members]),                 # Links
        #         "{:1.4f}".format(np.mean([l.weight for m in s.members for l in m.links])),    # Avg. weight
        #         # "{:1.4f}".format(np.mean([n.bias for m in s.members for n in m.neurons])),    # Avg. bias
        #         "{:1.4f}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),    # Avg. compatiblity
        #         s.numToSpawn])                                              # Nr. of members to spawn
        #
        # print(table)

        # phenotypes = hyperneat.epoch(SpeciesUpdate(fitnesses))
        phenotypes = hyperneat.epoch(MapElitesUpdate(fitnesses, pred))

    env.close()
