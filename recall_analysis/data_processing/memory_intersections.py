#%%
from itertools import combinations

import numpy as np
import pandas as pd

import paths
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
##
CONT_RESULTS_DIR = os.path.join(RESULTS_DIR, "forward_contiguity")
##
NOISE_RESULTS_DIR = os.path.join(RESULTS_DIR, "noise")
##
SEED_RESULTS_DIR = os.path.join(RESULTS_DIR, "seed")
###
POPULATION_SIZES_DIR = os.path.join(SEED_RESULTS_DIR, "population_sizes")
###
POPULATIONS_MEMORIES_DIR = os.path.join(SEED_RESULTS_DIR, "populations_memories")
###
RECALLS_SEED_DIR = os.path.join(SEED_RESULTS_DIR, "recalls")

#%%

np.seterr(all="raise")


# === Parameters ===
# Architecture
NUM_NEURONS = 10 ** 5
NUM_MEMORIES = 16
# Hebbian rule
SPARSITY = 0.1
# ==================


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

    return pop, neurons_per_pop
    # np.save(os.path.join("results", f"s{seed}-neurons-encoding"), neurons_encoding_mem)


from tqdm import tqdm
def get_memory_data():
    seeds = np.load("seeds.npy")
    seeds = seeds[:105]
    for seed in tqdm(seeds):
        (pops), (neurons_per_pop) = main(seed)
        np.save(
            os.path.join(POPULATIONS_MEMORIES_DIR, f"s{seed}-populations-memories"),
            pops,
        )
        np.save(
            os.path.join(POPULATION_SIZES_DIR, f"s{seed}-population-sizes"),
            neurons_per_pop,
        )
    return (
        pd.DataFrame(pops),
        pd.Series(neurons_per_pop),
    )


#%%

# Data load
((pops), (neurons_per_pop)) = get_memory_data()

#%%

def make_all_possible_intersections():

    # Get memory indices
    num_memories = pops.shape[1]
    memories_idx = np.arange(num_memories)
    # Make all possible combinations of two for each memory with another
    possible_transitions = [
        transition for transition in combinations(memories_idx, r=2)
    ]

    # Sum populations corresponding to each memory for combination of memories; dict ready for data frame
    intersections_dict = {
        transition: pops.loc[:, transition].sum(axis=1)
        for transition in possible_transitions
    }

    # Make data frames from dict
    intersections_data_frames = pd.DataFrame.from_dict(intersections_dict)

    # Mask the intersections (value of two) and set them to one
    intersections_data_frames = (intersections_data_frames == 2).astype(int)

    return intersections_data_frames


intersections_df = make_all_possible_intersections()

#%%


def get_intersection_sizes(intersections_data_frame):
    """
    Get the size of every possible memory intersection.

    :returns : pandas Series
        Multi-indexed with the combination of memory indices making a transition
    """

    # Shape neurons per pop as the intersection data frame ([populations, combinations])
    neurons_per_pop_array = np.tile(
        neurons_per_pop, (intersections_data_frame.shape[1], 1)
    )
    neurons_per_pop_array = np.rot90(neurons_per_pop_array, k=3)
    # Multiply by the size of the population (index)
    intersections_with_sizes_pop = intersections_data_frame.multiply(
        neurons_per_pop_array
    )
    # Sum along index to obtain the size of intersections
    intersection_sizes = intersections_with_sizes_pop.sum(axis=0)

    return pd.Series(intersection_sizes, name="intersection_sizes")


intersection_sizes = get_intersection_sizes(intersections_df)
