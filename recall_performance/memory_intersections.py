#%%
import os

import numpy as np

import generate_seeds

FIG_DIR = "fig"

os.makedirs(FIG_DIR, exist_ok=True)

np.seterr(all="raise")

generate_seeds.generate_seeds()

# === Parameters ===
# Architecture
NUM_NEURONS = 10 ** 5
NUM_MEMORIES = 16
# Hebbian rule
SPARSITY = 0.1
T_CYCLES = 100.0  # 14
T_STEP = 0.001
# Noise
NOISE_VAR = 65
# Initialization
FIRST_MEMORY = 7
# Recall
RECALL_THRESHOLD = 15


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


def main(arg):
    seed = arg

    np.random.seed(seed)

    # Compute memory patterns
    memory_patterns = np.random.choice(
        [0, 1], p=[1 - SPARSITY, SPARSITY], size=(NUM_NEURONS, NUM_MEMORIES)
    )

    # Compute populations
    pop, neurons_per_pop = np.unique(
        [tuple(i) for i in memory_patterns], axis=0, return_counts=True
    )

    num_pops = len(pop)

    population_num_encoding_mem = [
        (pop[:, memory] == 1).nonzero()[0] for memory in range(NUM_MEMORIES)
    ]
    return pop, neurons_per_pop, population_num_encoding_mem
    # np.save(os.path.join("results", f"s{seed}-neurons-encoding"), neurons_encoding_mem)


################
#%%
import pandas as pd


def get_memory_data():
    seeds = np.load("seeds.npy")
    seeds = seeds[:1]
    for seed in seeds:
        (pops), (neurons_per_pop), (pop_num_encoding_mem) = main(seed)
    return (
        pd.DataFrame(pops),
        pd.DataFrame(neurons_per_pop),
        pd.DataFrame(pop_num_encoding_mem),
    )

