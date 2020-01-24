"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Detects when the firing rate of a memory crosses a recall gain_threshold,
and is therefore recalled
"""

from itertools import combinations

import numpy as np


def count_memory_recalls(firing_rates, recall_threshold):
    """ THIS IS WRONGO """

    shape_array = firing_rates.shape
    recall_counts = np.zeros(shape_array[0], dtype=int)

    # iterate over firing rates of each memory at time step
    for memory_idx in range(shape_array[0]):
        for time_step in range(shape_array[1]):
            if time_step == 0:
                continue

            # If increasing and surpassing gain_threshold, increase count
            if (firing_rates[memory_idx, time_step - 1] < recall_threshold) and (
                firing_rates[memory_idx, time_step] > recall_threshold
            ):
                recall_counts[memory_idx] += 1

    return recall_counts


def get_memory_recalls(recall_threshold, firing_rates, time_step):

    firing_rates_recalls = np.int_(firing_rates > recall_threshold)
    cycle_indices = np.arange(start=0, end=firing_rates.shape[1], step=1 / time_step)
    recalls = firing_rates_recalls[:, cycle_indices]

    return recalls


def get_probability_recall(recalls_per_memory, t_cycles):
    return np.array([(recalls / t_cycles) for recalls in recalls_per_memory])


def get_memory_intersections(memory_patterns):
    r = 2
    memory_combinations_keys = list(
        combinations(np.arange(memory_patterns.shape[1]), r)
    )
    memory_combinations = list(
        combinations(
            [memory_patterns[:, p] for p in range(memory_patterns.shape[1])], r
        )
    )

    memory_intersections = [
        np.multiply(*combination) for combination in memory_combinations
    ]

    return {
        memory_combinations_key: memory_intersection
        for memory_combinations_key in memory_combinations_keys
        for memory_intersection in memory_intersections
    }


def get_memory_intersection_sizes(memory_intersections):
    return {
        memory_combinations_keys: np.sum(memory_intersection)
        for memory_combinations_keys in memory_intersections.keys()
        for memory_intersection in memory_intersections.values()
    }

def check_only_single_recall(binary_recalls):
    assert len(binary_recalls.shape) == 2
    array_sum = np.sum(np.sum(binary_recalls))
    assert array_sum <= binary_recalls.shape[0]

def get_memory_jumps(recalls):

    check_only_single_recall(recalls)
    # Get a column array of the memory indexes
    memory_identifier = np.rot90(np.arange(recalls.shape[0]))
    # Multiply binary array by previous to identify the memories, flatten
    recalled_memories_time_list = np.sum((recalls * memory_identifier), axis=1)
    # Substract one element from the previous to get the size of memory jumps
    memory_jump_sizes = np.diff(recalled_memories_time_list)

    return memory_jump_sizes


def get_inter_retrieval_times(memory_jumps):
    """
    IRT is the number of iterations that pass before a new memory is recalled.
    e.g. p1, p1, p1, p5; IRT = 3
    """

    # There is no jump at the very beginning
    first_jump_idx = [-1]
    # Memories jump where difference of recalled memory is not 0; get index
    where_jumps = np.nonzero(memory_jumps)
    # Stack "real first jump" to compute diffs correctly
    where_jumps_probed = np.hstack((first_jump_idx, where_jumps))
    # Jump after one iteration is IRT=1, not IRT=0
    where_jumps_probed += 1
    # Diffs of jump indexes gives iterations until jump == IRT
    inter_retrieval_times = np.diff(where_jumps_probed)

    return inter_retrieval_times
