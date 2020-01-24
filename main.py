"""
<[Re] Recanatesi (2015). Neural Network Model of Memory Retrieval>
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import pickle
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

import tools.plots as plot
import tools.recall_performance
import tools.sine_wave

FIG_DIR = "fig"

os.makedirs(FIG_DIR, exist_ok=True)

np.seterr(all="raise")


# === Parameters ===
# Architecture
NUM_NEURONS = 10 ** 5
NUM_MEMORIES = 16
# Activation
T_DECAY = 0.01
# Gain
GAIN_THRESHOLD = 0
GAIN_EXP = 2 / 5
# Hebbian rule
EXCITATION = 13_000
SPARSITY = 0.1
# Inhibition
SIN_MIN = 0.7
SIN_MAX = 1.06
T_OSCILLATION = 1
PHASE_SHIFT = 0.75
# Short term association
CONT_FORTH = 1500
CONT_BACK = 400
# Time
T_CYCLES = 14.0  # 14
T_STEP = 0.001
# Noise
NOISE_VAR = 65
# Initialization
INIT_RATE = 1
FIRST_MEMORY = 7
# Recall
RECALL_THRESHOLD = 15
# Replication parameters
PARAM_NOISE = 700
# 10 we see saw peaks and bubbles and good transitions
# print(param_noise)
PARAM_INHIBITION = 1
# print(param_inhibition)
PARAM_CURRENT = 1  # 4.75

# ==================


def inside_loop_connectivity_matrices(arg):
    """ Compute the loop inside connectivities to parallelize """

    i, num_pops, pop, backward_cont, forward_cont = arg

    regular = np.zeros(num_pops)
    forward = np.zeros(num_pops)
    backward = np.zeros(num_pops)

    for j in range(num_pops):
        regular[j] = np.sum((pop[i, :] - SPARSITY) * (pop[j, :] - SPARSITY))

        forward[j] = np.sum(pop[i, forward_cont] * pop[j, forward_cont + 1])

        backward[j] = np.sum(pop[i, backward_cont] * pop[j, backward_cont - 1])

    return regular, forward, backward


def compute_connectivity_matrices(num_pops, pop, forward_cont, backward_cont):
    """ Compute matrices with parallelization """

    regular_connectivity = np.zeros((num_pops, num_pops))
    forward_connectivity = np.zeros((num_pops, num_pops))
    backward_connectivity = np.zeros((num_pops, num_pops))

    args = [(i, num_pops, pop, backward_cont, forward_cont) for i in range(num_pops)]

    with Pool(cpu_count()) as pool:
        pools = list(
            tqdm(pool.imap(inside_loop_connectivity_matrices, args), total=len(args))
        )
        pool.close()
        pool.join()

    for i in range(num_pops):
        regular, forward, backward = pools[i]
        regular_connectivity[i, :] = regular
        forward_connectivity[i, :] = forward
        backward_connectivity[i, :] = backward

    return regular_connectivity, forward_connectivity, backward_connectivity


def main():
    """ Licensing and main execution """

    print(
        "<[Re] Recanatesi (2015)>  Copyright (C) <2019>\n"
        "<de la Torre-Ortiz C, Nioche A>\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the 'LICENSE' file for details.\n"
    )

    np.random.seed(123)

    # Compute memory patterns
    print("Computing memory patterns and neuron populations...")
    memory_patterns = np.random.choice(
        [0, 1], p=[1 - SPARSITY, SPARSITY], size=(NUM_NEURONS, NUM_MEMORIES)
    )

    # Compute populations
    pop, neurons_per_pop = np.unique(
        [tuple(i) for i in memory_patterns], axis=0, return_counts=True
    )

    num_pops = len(pop)

    neurons_encoding_mem = [
        (pop[:, mu] == 1).nonzero()[0] for mu in range(NUM_MEMORIES)
    ]

    # === Other pre-computations ===
    num_iter = int(T_CYCLES / T_STEP)
    # relative_excitation = excitation / NUM_NEURONS
    time_param = T_STEP / T_DECAY

    # Inhibition
    sine_wave = np.zeros(num_iter)

    # Connectivity
    forward_cont = np.arange(NUM_MEMORIES - 1)
    backward_cont = np.arange(1, NUM_MEMORIES)

    # Noise
    noise = np.zeros((num_pops, num_iter))

    # Neuron dynamics
    firing_rates = np.zeros([num_pops])
    current = np.zeros([num_pops])
    average_firing_rates_per_memory = np.zeros((NUM_MEMORIES, num_iter))
    currents = np.zeros((num_pops, num_iter))
    currents_memory = np.zeros((NUM_MEMORIES, num_iter))
    # ==============================

    # Compute sine wave
    print("Calculating inhibition values...")

    for time in range(num_iter):
        sine_wave[time] = tools.sine_wave.sinusoid(
            min_=SIN_MIN,
            max_=SIN_MAX,
            period=T_OSCILLATION,
            t=time,
            phase_shift=PHASE_SHIFT * T_OSCILLATION,
            dt=T_STEP,
        )

    inhibition = -sine_wave * PARAM_INHIBITION

    # Compute weights
    print("Computing regular, forward and backward connectivity...")

    bkp_file = "bkp/connectivity.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file):
        connectivities = compute_connectivity_matrices(
            num_pops, pop, forward_cont, backward_cont
        )
        pickle.dump(connectivities, open(bkp_file, "wb"))

    else:
        print("Loading connectivity from pickle file...")
        connectivities = pickle.load(open(bkp_file, "rb"))

    # regular_connectivity, forward_connectivity, backward_connectivity = \
    #     compute_connectivity_matrices(num_pops, pop, forward_cont,
    #                                   backward_cont)
    regular_connectivity, forward_connectivity, backward_connectivity = connectivities

    regular_connectivity *= EXCITATION  # MOD REMOVED / NUM_NEURONS
    forward_connectivity *= CONT_FORTH
    backward_connectivity *= CONT_BACK
    inhibition *= EXCITATION  # MOD REMOVED / NUM_NEURONS

    weights_without_inhibition = (
        regular_connectivity + forward_connectivity + backward_connectivity
    )

    # Compute noise
    print("Calculating uncorrelated Gaussian noise...")

    for pop in range(num_pops):
        noise[pop] = (
            np.random.normal(
                loc=0, scale=(NOISE_VAR * neurons_per_pop[pop]) ** 0.5, size=num_iter
            )
            / neurons_per_pop[pop]
            * PARAM_NOISE
        )  # MOD ADDED BACK

    # Initialize firing rates
    firing_rates[neurons_encoding_mem[FIRST_MEMORY]] = INIT_RATE

    # Initialize current
    c_ini = INIT_RATE ** (1 / GAIN_EXP) - GAIN_THRESHOLD
    current[neurons_encoding_mem[FIRST_MEMORY]] = c_ini

    bkp_file = "bkp/average_firing_rates_per_memory.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file):
        # Compute neuron dynamics
        print("Computing dynamics at each time step...")

        for time in tqdm(range(num_iter)):

            # Update current
            for pop in range(num_pops):
                # Compute weights
                weights = (
                    weights_without_inhibition[pop, :] + inhibition[time]
                )  # / NUM_NEURONS

                # Compute input     CHANGE neur to neurons_per_pop[:]
                s_v = neurons_per_pop[:] / NUM_NEURONS  # DID NOT CHANGE
                input_v = np.sum(weights[:] * s_v * firing_rates[:])

                current[pop] += time_param * (
                    -current[pop] + input_v + noise[pop, time]
                )

                # Backup for the plot
                currents[pop, time] = current[pop]

            # Update firing rates
            firing_rates[:] = 0
            cond = (current + GAIN_THRESHOLD) > 0
            firing_rates[cond] = (
                current[cond] * PARAM_CURRENT + GAIN_THRESHOLD
            ) ** GAIN_EXP

            for memory_idx in range(NUM_MEMORIES):
                f_r = firing_rates[neurons_encoding_mem[memory_idx]]
                n_corresponding = neurons_per_pop[neurons_encoding_mem[memory_idx]]

                average_firing_rates_per_memory[memory_idx, time] = np.average(
                    f_r, weights=n_corresponding
                )

                c_mu = current[neurons_encoding_mem[memory_idx]]

                currents_memory[memory_idx, time] = np.average(
                    c_mu, weights=n_corresponding
                )

        pickle.dump(average_firing_rates_per_memory, open(bkp_file, "wb"))

    else:
        print("Loading connectivity from pickle file...")
        average_firing_rates_per_memory = pickle.load(open(bkp_file, "rb"))

    # Recall performance
    counts_memory_recalls = tools.recall_performance.count_memory_recalls(
        average_firing_rates_per_memory, RECALL_THRESHOLD
    )
    probability_recall_memories = tools.recall_performance.get_probability_recall(
        counts_memory_recalls, T_CYCLES
    )

    # Plots
    print("Plotting...")
    # plot.currents(currents, dt=t_step,
    #               type_="population", fig_num=0)
    # plot.currents(currents_memory, dt=t_step,
    #               type_="memory", fig_num=1)

    plot.firing_rates(average_firing_rates_per_memory, dt=T_STEP)
    plot.attractors(average_firing_rates_per_memory, dt=T_STEP)

    # plot.sine_wave(sine_wave, dt=t_step)
    # plot.inhibition(inhibition, dt=t_step)

    # plot.weights(weights_without_inhibition,
    #              type_="no_inhibition", fig_num=0)
    # plot.weights(regular_connectivity,
    #              type_="regular", fig_num=1)
    # plot.weights(forward_connectivity,
    #              type_="forward", fig_num=2)
    # plot.weights(backward_connectivity,
    #              type_="backward", fig_num=3)

    # plot.noise(noise, dt=t_step, t_tot=t_tot)

    plot.probability_recall_given_size(
        neurons_encoding_mem, probability_recall_memories
    )

    print("Done!")


if __name__ == "__main__":
    main()
