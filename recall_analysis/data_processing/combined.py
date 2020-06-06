#%%
"""

"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_memories_transition_before_after(recalls_analysis_data_frame):
    """ When recalled memory changes to another, get the previous and next memories """

    after_jumps = recalls_analysis_data_frame["memory_recalled"][
        recalls_analysis_data_frame["transition"] == 1
    ]
    after_jumps_idx = after_jumps.index

    before_jumps_idx = after_jumps_idx - 1
    before_jumps = recalls_analysis_data_frame.loc[before_jumps_idx, "memory_recalled"]
    return before_jumps, after_jumps


def get_transitions_memories(before_jumps, after_jumps):
    return pd.Series(
        [sorted(transition) for transition in zip(before_jumps, after_jumps)],
        name="transition_memories",
    )


def get_transitions_sizes(intersection_sizes, transitions_memories):
    return pd.Series(
        [
            intersection_sizes.loc[tuple(memory_pair)]
            for memory_pair in transitions_memories
        ]
    )


def sync_index_to_master_data_frame(transitions_memories, transition_sizes, master):
    transitions_memories.index = transition_sizes.index = master.index


def make_all(recalls_analysis_data_frames, intersection_sizes_all):

    assert len(recalls_analysis_data_frames) == len(intersection_sizes_all)

    print("Building all series for memory intersection size...")
    for idx in tqdm(range(len(recalls_analysis_data_frames))):

        before_jumps, after_jumps = get_memories_transition_before_after(
            recalls_analysis_data_frames[idx]
        )
        # Use recalls data to get the memories before and after transition
        transitions_memories = get_transitions_memories(before_jumps, after_jumps)
        # Use memory intersections to get transition sizes
        transition_sizes = get_transitions_sizes(
            intersection_sizes_all[idx], transitions_memories
        )
        # Make indices the same to be able to merge them to main data frame
        sync_index_to_master_data_frame(
            transitions_memories, transition_sizes, after_jumps
        )
        # Add all previously calculated columns to main data frame
        recalls_analysis_data_frames[idx]["transition_memories"] = transitions_memories
        recalls_analysis_data_frames[idx]["transition_size"] = transition_sizes
        # Change NaN for 0 as they always happen from some transitions methods
        recalls_analysis_data_frames[idx] = recalls_analysis_data_frames[idx].fillna(0)

    return recalls_analysis_data_frames

