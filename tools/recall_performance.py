"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Detects when the firing rate of a memory crosses a recall gain_threshold,
and is therefore recalled
"""

import numpy as np
from itertools import combinations


def count_memory_recalls(firing_rates, recall_threshold):
    """"""

    shape_array = firing_rates.shape
    recall_counts = np.zeros(shape_array[0], dtype=int)

    # iterate over firing rates of each memory at time step
    for memory_idx in range(shape_array[0]):
        for time_step in range(shape_array[1]):
            if time_step == 0:
                continue

            # If increasing and surpassing gain_threshold, increase count
            if (firing_rates[
                    memory_idx, time_step - 1] < recall_threshold) and (
                    firing_rates[memory_idx, time_step] > recall_threshold):
                recall_counts[memory_idx] += 1

    return recall_counts


def get_probability_recall(recalls_per_memory, t_cycles):
    return np.array([(recalls / t_cycles) for recalls in
                     recalls_per_memory])


def get_memory_intersections(memory_patterns):
    r = 2
    memory_combinations_keys = list(
        combinations(np.arange(memory_patterns.shape[1]), r))
    memory_combinations = list(combinations(
        [memory_patterns[:, p] for p in range(memory_patterns.shape[1])], r))

    memory_intersections = [np.multiply(*combination) for combination in
                            memory_combinations]

    return {memory_combinations_key: memory_intersection for
            memory_combinations_key in memory_combinations_keys for
            memory_intersection in memory_intersections}


def get_memory_intersection_sizes(memory_intersections):
    return {memory_combinations_keys: np.sum(memory_intersection) for
            memory_combinations_keys in memory_intersections.keys() for
            memory_intersection in memory_intersections.values()}
