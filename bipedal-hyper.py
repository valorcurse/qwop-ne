from typing import List, Dict, Any

import argparse
import os
import sys
import psutil

import multiprocessing as mp

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

env_name = "BipedalWalker-v2"
nproc = 4

envs_size = 50
pop_size = 100
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

def test_organism(phenotypes, envs):
    feedforward = FeedforwardCUDA(phenotypes)

    observations = envs.reset()

    obs_32 = np.float32(observations)
    actions = feedforward.update(obs_32)

    fitnesses = np.zeros(len(envs.remotes), dtype=np.float64)

    done = False
    done_tracker = np.array([False for _ in range(len(envs.remotes))])

    diff = len(phenotypes) - len(envs.remotes)
    if diff < 0:
        done_tracker[diff:] = True

    distances = np.zeros(len(envs.remotes))
    last_distances = []
    stagnations = np.zeros(len(envs.remotes))

    all_states = []


    max_steps = 50
    steps = max_steps
    while not done:
        actions = np.pad(actions, (0, abs(diff)), 'constant')
        states, rewards, dones, info = envs.step(actions)
        actions = feedforward.update(states)

        fitnesses[done_tracker == False] += np.around(rewards[done_tracker == False], decimals=2)
        # fitnesses[done_tracker == False] = np.around(rewards[done_tracker == False], decimals=2)

        envs_done = dones == True
        done_tracker[envs_done] = dones[envs_done]
        envs_running = len([d for d in done_tracker if d == False])

        # print(" " * 100, end='\r', flush=True)
        # print("Envs running: {}/{}".format(envs_running, len(phenotypes)), end='\r')

        # print(done_tracker)
        # print(fitnesses)

        done = envs_running == 0

        distances += states.T[2]

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

def chunk_testing(phenotypes):
    all_states = []
    fitnesses = []
    for chunk in chunks(phenotypes, envs_size):
        fitness, states = test_organism(chunk, envs)
        all_states.extend(states)
        fitnesses.extend(fitness)

    return (np.array(all_states), np.array(fitnesses))

def run_env_once(phenotype):
    env = gym.make(env_name)
    # Visualize().update(phenotype)
    feedforward_highest = FeedforwardCUDA([phenotype])
    states = env.reset()
    done = False
    while not done:
        actions = feedforward_highest.update(np.array([states]))
        states, reward, done, info = env.step(actions[0])
        env.render()
    env.close()
    # Visualize().close()

def refine_ae(autoencoder, phenotypes):
    autoencoder.load_weights('basic.h5')

    last_loss = 1000.0
    loss = 1000.0

    while loss <= last_loss:
        last_loss = loss

        states, _ = chunk_testing(phenotypes)

        ten_percent = max(1, int(states.shape[0] * 0.1))

        train = states[:-ten_percent]
        test = states[-ten_percent:]

        hist = autoencoder.fit(train, train,
                               epochs=50,
                               batch_size=256,
                               shuffle=True,
                               validation_data=(test, test),
                               verbose=0)

        loss = abs(hist.history['val_loss'][-1])

        print("Training autoencoder. Loss: {}".format(loss))

def pad_matrix(all_states, matrix_width):
    padded = []
    for row in all_states:
        # row = all_states[:, i].flatten()

        row = np.pad(row, (0, abs(matrix_width - row.shape[0])), 'constant')
        padded.append(row)

    return np.array(padded)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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

    # this is our input placeholder
    input_img = Input(shape=(behavior_matrix_size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(behavior_matrix_size, activation='relu')(encoded)

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
    autoencoder.save_weights('basic.h5')
    ############################################################################################################

    print("Creating hyperneat object")
    # pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    pop_config = MapElitesConfiguration(4, pop_size, encoding_dim, inputs, outputs)
    # hyperneat = hn.HyperNEAT(pop_config)
    hyperneat = hn.NEAT(pop_config)

    start_fitness = [0.0]*pop_size
    start_features = np.zeros((pop_size, encoding_dim))
    phenotypes: List[Phenotype] = hyperneat.epoch(MapElitesUpdate(start_fitness, start_features))

    highestFitness: float = -1000.0
    highestDistance: float = 0.0

    envs = [make_env(env_name, seed) for seed in range(envs_size)]

    print("Creating envs...")
    envs = SubprocVecEnv(envs)
    print("Done.")

    loss = 1000.0
    max_stagnation = 50
    progress_stagnation = 0
    train_ae = progress_stagnation == max_stagnation

    refine_ae(autoencoder, phenotypes)

    epoch_num = 0
    while True:
        epoch_num += 1
        train_ae = progress_stagnation == max_stagnation

        if train_ae:
            ae_phenotypes = []
            if len(hyperneat.population.archive) > 0:

                sorted_archive = sorted(hyperneat.population.archive.values(), key=lambda a: a['fitness'])
                ae_phenotypes = [g['genome'].createPhenotype() for g in sorted_archive]
            else:
                ae_phenotypes = phenotypes

            refine_ae(autoencoder, ae_phenotypes)
            ten_percent = max(1, int(len(ae_phenotypes) * 0.1))
            phenotypes = ae_phenotypes[:ten_percent]

            hyperneat.population.archive = {}
            progress_stagnation = 0


        print("########## Epoch {} ##########".format(epoch_num))

        # Test the phenotypes in the envs
        start = time.time()
        all_states, fitnesses = chunk_testing(phenotypes)
        end = time.time()
        print("Time:", end - start)


        pred = encoder.predict(all_states)

        print("Highest fitness this epoch:", max(fitnesses))
        max_fitness = max(zip(fitnesses, phenotypes), key=lambda e: e[0])

        # mpc = mp.get_context('spawn')
        # p = mpc.Process(target=run_env_once, args=(max_fitness[1],))
        # p.start()

        if max_fitness[0] > highestFitness:
            print("New highest fitness: {}".format(max_fitness))
            highestFitness = max_fitness[0]

            best_phenotype = max_fitness[1]
            # Visualize().update(best_phenotype)
            run_env_once(best_phenotype)
            # Visualize().close()

            progress_stagnation = 0
        else:
            progress_stagnation += 1

        print("Highest fitness all-time: {}".format(highestFitness))
        print("Progress stagnation: {}".format(progress_stagnation))

        total = pow(hyperneat.population.configuration.mapResolution,
                    hyperneat.population.configuration.features)
        archiveFilled = len(hyperneat.population.archivedGenomes) / total
        print("Genomes in archive: {}".format(len(hyperneat.population.archive)))

        archive_size = max(1, len(hyperneat.population.archive))
        avg_neurons = sum([len(g['genome'].neurons) for g in hyperneat.population.archive.values()]) / archive_size - (inputs+outputs)
        avg_links = sum([len(g['genome'].links) for g in hyperneat.population.archive.values()]) / archive_size
        avg_fitness = sum([g['fitness'] for g in hyperneat.population.archive.values()])/max(1, len(hyperneat.population.archive))

        table = PrettyTable(["Epoch", "fitness", "max fitness", "neurons", "links", "avg. fitness", "archive"])
        table.add_row([
            epoch_num,
            "{:1.4f}".format(max_fitness[0]),
            "{:1.4f}".format(highestFitness),
            "{:1.4f}".format(avg_neurons),
            "{:1.4f}".format(avg_links),
            "{:1.4f}".format(avg_fitness),
            "{:1.8f}".format(archiveFilled)])
        print(table)

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
