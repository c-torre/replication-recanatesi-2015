#%%
"""

"""

from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

import paths
from recall_analysis.data_processing.pre_processing import loader_utils

# Data load
# ((pops), (neurons_per_pop)) = get_memory_data()

#%%


def make_all_possible_intersections(populations_memories):

    # Get memory indices
    num_memories = populations_memories.shape[1]
    memories_idx = np.arange(num_memories)
    # Make all possible combinations of two for each memory with another
    possible_transitions = [
        transition for transition in combinations(memories_idx, r=2)
    ]

    # Sum populations corresponding to each memory for combination of memories; dict ready for data frame
    intersections_dict = {
        transition: populations_memories.loc[:, transition].sum(axis=1)
        for transition in possible_transitions
    }

    # Make data frames from dict
    intersections_data_frames = pd.DataFrame.from_dict(intersections_dict)

    # Mask the intersections (value of two) and set them to one
    intersections_data_frames = (intersections_data_frames == 2).astype(int)

    return intersections_data_frames


# intersections_df = make_all_possible_intersections(populations_memories[0])

#%%


def get_intersection_sizes(intersections_data_frame, population_sizes):
    """
    Get the size of every possible memory intersection.

    :returns : pandas Series
        Multi-indexed with the combination of memory indices making a transition
    """

    # Shape neurons per pop as the intersection data frame ([populations, combinations])
    neurons_per_pop_array = np.tile(
        population_sizes, (intersections_data_frame.shape[1], 1)
    )
    neurons_per_pop_array = np.rot90(neurons_per_pop_array, k=3)
    # Multiply by the size of the population (index)
    intersections_with_sizes_pop = intersections_data_frame.multiply(
        neurons_per_pop_array
    )
    # Sum along index to obtain the size of intersections
    intersection_sizes = intersections_with_sizes_pop.sum(axis=0)

    return pd.Series(intersection_sizes, name="intersection_sizes")


# intersection_sizes = get_intersection_sizes(intersections_df)

#%%


def make_all():

    populations_memories_dir = paths.POPULATIONS_MEMORIES_DIR
    population_sizes_dir = paths.POPULATION_SIZES_DIR

    populations_memories = loader_utils.get_arrays_from_files(
        populations_memories_dir, calling_intersections=True
    )
    # print(type(populations_memories))
    # print(list(populations_memories.values()))
    populations_memories = loader_utils.arrays_to_data_frames(populations_memories)
    population_sizes = loader_utils.get_arrays_from_files(
        population_sizes_dir, calling_intersections=True
    )

    assert len(populations_memories) == len(population_sizes)

    intersection_sizes_all = []

    print("Building all series for memory intersection size...")
    for idx in tqdm(range(len(population_sizes))):
        intersections_data_frame = make_all_possible_intersections(
            populations_memories[idx]
        )
        intersections_sizes = get_intersection_sizes(
            intersections_data_frame, population_sizes[idx]
        )
        intersection_sizes_all.append(intersections_sizes)

    assert intersection_sizes_all

    return intersection_sizes_all


if __name__ == "__main__":
    intersections_sizes_all = make_all()
