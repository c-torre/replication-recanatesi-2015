"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Detects when the firing rate of a memory crosses a recall threshold,
and is therefore recalled
"""

import numpy as np


def detect_memory_recalls(firing_rates, recall_threshold):
    """"""

    shape_array = firing_rates.shape
    recall_counts = np.zeros(shape_array[1])

    for memory_idx in range(shape_array[0]):
        for time_step in range(shape_array[1]):
            if time_step == 0:
                continue

            if (firing_rates[
                    memory_idx, time_step - 1] < recall_threshold) and (
                    firing_rates[memory_idx, time_step] > recall_threshold):
                recall_counts[memory_idx] += 1

    return recall_counts


def get_probability_recall(recalls_per_memory):
    return np.array([(recalls / np.amax(recalls_per_memory)) for recalls in
                     recalls_per_memory])
