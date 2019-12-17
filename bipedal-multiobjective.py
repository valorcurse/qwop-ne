from typing import Tuple, List


if __name__ == '__main__':

    import gym
    import numpy as np
    import pyglet
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    from prettytable import PrettyTable

    # import neat.hyperneat as hn
    import neat.visualize as dash_vis
    from neat.neat import NEAT, Evaluation
    from neat.neatTypes import NeuronType
    from neat.visualize import Visualize
    from neat.phenotypes import FeedforwardCUDA, Phenotype
    from neat.multiobjectivePopulation import MOConfiguration

    import time
    from multiprocessing import Queue
    import random

    env_name = "BipedalWalker-v2"
    nproc = 4

    envs_size = 90
    pop_size = 270
    max_stagnation = 25
    # encoding_dim = 8
    features_dimensions = 14
    behavior_steps = 25
    behavior_dimensions = features_dimensions * behavior_steps

    q = Queue()
    vis = Visualize()
    vis.start()

    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        return _f


    def run_env_once(phenotype, env):
        feedforward_highest = FeedforwardCUDA()
        states = env.reset()

        done = False
        distance = 0.0
        last_distance = 0.0
        distance_stagnation = 0

        image = env.render(mode='rgb_array')

        images = []
        activations = []
        while not done:
            actions = feedforward_highest.update([phenotype], np.array([states]))

            # print(actions)
            states, reward, done, info = env.step(actions[0])
            distance += np.around(states[2], decimals=2)

            if distance <= last_distance:
                distance_stagnation += 1
            else:
                distance_stagnation = 0

            if distance_stagnation >= 100:
                done = True

            last_distance = distance

            # activations.append(feedforward_highest.mem[0])
            # images.append(env.render(mode='rgb_array'))
            env.render()

        # dash_vis.dash_queue.put([phenotype.graph, activations, images])
        # norm_acts = (np.array(activations)+1.0)/2.0
        # dash_vis.dash_queue.put([phenotype.graph, activations])

    def pad_matrix(all_states, matrix_width):
        padded = []
        for row in all_states:
            # row = all_states[:, i].flatten()

            row = np.pad(row, (0, abs(matrix_width - row.shape[0])), 'constant')
            padded.append(row)

        return np.array(padded)

    class TestOrganism(Evaluation):

        def __init__(self):
            print("Creating envs...")
            self.envs = SubprocVecEnv([make_env(env_name, seed) for seed in range(envs_size)])
            self.num_of_envs = envs_size
            self.feedforward = FeedforwardCUDA()
            print("Done.")

        def evaluate(self, phenotypes: List[Phenotype]) -> Tuple[np.ndarray, np.ndarray]:

            # feedforward = FeedforwardCUDA(phenotypes)

            observations = self.envs.reset()

            obs_32 = np.float32(observations)
            actions = self.feedforward.update(phenotypes, obs_32)

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

            max_steps = 10
            steps = max_steps
            while not done:
                actions = np.pad(actions, (0, abs(diff)), 'constant')
                states, rewards, dones, info = self.envs.step(actions)


                actions = self.feedforward.update(phenotypes, states)

                # fitnesses[done_tracker == False] = np.around(rewards[done_tracker == False], decimals=2)
                fitnesses[done_tracker == False] += np.around(rewards[done_tracker == False], decimals=2)
                envs_done = dones == True
                done_tracker[envs_done] = dones[envs_done]
                envs_running = len([d for d in done_tracker if d == False])

                done = envs_running == 0

                distances += np.around(states.T[2], decimals=2)

                stagnations += distances == last_distances

                done_tracker[stagnations >= 100] = True

                last_distances = distances

                relevant_states = states[:, :features_dimensions]
                running_states = np.zeros(relevant_states.shape)
                running_states[done_tracker == False] += relevant_states[done_tracker == False]

                if steps == max_steps:
                    steps = 0
                    all_states.append(running_states)

                steps += 1

            all_states = np.array(all_states)
            flattened_states = []
            for row_i in range(all_states.shape[1]):
                flattened_states.append(all_states[:, row_i].flatten())

            flattened_states = pad_matrix(np.array(flattened_states), behavior_dimensions)
            return (fitnesses[:len(phenotypes)], flattened_states[:len(phenotypes)])

    env = gym.make(env_name)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    print("Creating neat object")
    obj_ranges = [(-100.0, 300.0), (-1.0, 1.0)]
    pop_config = MOConfiguration(pop_size, inputs, outputs, behavior_dimensions, obj_ranges)
    neat = NEAT(TestOrganism(), pop_config)

    highest_fitness = -1000.0

    start = time.time()
    for _ in neat.epoch():
        print("Epoch Time: {}".format(time.time() - start))
        random_phenotype = random.choice(neat.phenotypes)
        most_fit =  max([(g.fitness, g) for g in neat.population.genomes], key=lambda e: e[0])
        # most_novel =  max([(g.novelty, g) for g in neat.population.genomes], key=lambda e: e[0])
        # best_phenotype =  max(neat.phenotypes, key=lambda e: e.fitness)

        # if max_fitness[0] >= highest_fitness:
        #     run_env_once(max_fitness[1].createPhenotype(), env)
        #     highest_fitness = max_fitness[0]

        # run_env_once(most_novel[1].createPhenotype(), env)
        # run_env_once(most_fit[1].createPhenotype(), env)
        # run_env_once(random_phenotype, env)

        if most_fit[0] >= highest_fitness:
            run_env_once(most_fit[1].createPhenotype(), env)
            highest_fitness = most_fit[0]

        print("Highest fitness all-time: {}".format(highest_fitness))

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
                "{:1.2f}".format(np.mean([len([n for n in p.graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for p in neat.phenotypes])),
                # Links
                "{:1.2f}".format(np.mean([len(p.graph.edges) for p in neat.phenotypes])),
                # Avg. weight
                "{:1.2f}".format(np.mean([l.weight for m in s.members for l in m.links])),
                # Avg. compatiblity
                "{:1.2}".format(np.mean([m.calculateCompatibilityDistance(s.leader) for m in s.members])),
                # Nr. of members to spawn
                s.numToSpawn])

        print(table)

        start = time.time()
        print("########## Epoch {} ##########".format(neat.epochs))

    env.close()





