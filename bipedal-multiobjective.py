from typing import Tuple


if __name__ == '__main__':

    import gym
    import numpy as np
    import pyglet
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    from prettytable import PrettyTable

    # import neat.hyperneat as hn
    from neat.neat import NEAT, Evaluation
    from neat.neatTypes import NeuronType
    from neat.phenotypes import FeedforwardCUDA
    from neat.multiobjectivePopulation import MOConfiguration

    import time

    env_name = "BipedalWalker-v2"
    nproc = 4

    envs_size = 80
    pop_size = 80
    max_stagnation = 25
    encoding_dim = 8
    behavior_dimensions = 14
    behavior_steps = 250
    behavior_matrix_size = behavior_dimensions * behavior_steps


    class DrawText:
        def __init__(self, label: pyglet.text.Label):
            self.label = label

        def render(self):
            self.label.draw()

    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        return _f


    def run_env_once(phenotype, env):
        feedforward_highest = FeedforwardCUDA([phenotype])
        states = env.reset()

        # env.render()
        # label = pyglet.text.Label("test", font_size=36,
        #                           x=50, y=50, anchor_x='left', anchor_y='bottom',
        #                           color=(255, 123, 255, 255))
        # pyglabel = DrawText(label)
        # env.viewer.add_geom(pyglabel)

        done = False
        distance = 0.0
        last_distance = 0.0
        distance_stagnation = 0


        while not done:
            actions = feedforward_highest.update(np.array([states]))

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


            # pyglabel.label.text = str(actions)
            # label.draw()

            env.render()


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

            # all_states = np.empty((1, 14))
            all_states = []

            max_steps = 10
            steps = max_steps
            while not done:
                actions = np.pad(actions, (0, abs(diff)), 'constant')
                states, rewards, dones, info = self.envs.step(actions)

                # if render:
                #     envs.remotes[0].send(('render', None))
                #     envs.remotes[0].recv()

                actions = feedforward.update(states)

                # fitnesses[done_tracker == False] += np.around(rewards[done_tracker == False], decimals=2)
                fitnesses[done_tracker == False] = np.around(rewards[done_tracker == False], decimals=2)

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

                relevant_states = states[:, :14]
                running_states = np.zeros(relevant_states.shape)
                running_states[done_tracker == False] += relevant_states[done_tracker == False]

                if steps == max_steps:
                    steps = 0
                    all_states.append(running_states)
                    # all_states = np.vstack((all_states, states[:, :14]))
                    # print(all_states.shape)

                steps += 1

            all_states = np.array(all_states)
            flattened_states = []
            for row_i in range(all_states.shape[1]):
                flattened_states.append(all_states[:, row_i].flatten())

            flattened_states = pad_matrix(np.array(flattened_states), behavior_matrix_size)

            return (fitnesses[:len(phenotypes)], flattened_states[:len(phenotypes)])

    env = gym.make(env_name)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    print("Creating neat object")
    pop_config = MOConfiguration(pop_size, inputs, outputs)
    neat = NEAT(TestOrganism(), pop_config)

    highest_fitness = -1000.0

    start = time.time()
    for _ in neat.epoch():
        print("Epoch Time: {}".format(time.time() - start))
        max_fitness =  max([(g.fitness, g) for g in neat.population.genomes], key=lambda e: e[0])

        # if max_fitness[0] >= highest_fitness:
        #     run_env_once(max_fitness[1].createPhenotype(), env)
        #     highest_fitness = max_fitness[0]

        run_env_once(max_fitness[1].createPhenotype(), env)
        if max_fitness[0] >= highest_fitness:
            highest_fitness = max_fitness[0]

        print("Highest fitness all-time: {}".format(highest_fitness))
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
                "{:1.2f}".format(np.mean([len([n for n in m.neurons if n.neuronType == NeuronType.HIDDEN]) for m in s.members])),
                # Links
                "{:1.2f}".format(np.mean([len(m.links) for m in s.members])),
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




