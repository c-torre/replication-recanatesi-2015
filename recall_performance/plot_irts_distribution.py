#%%
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from recall_performance import npy_loader_cont_forth

PLOTS_DIR = "plt"
RESULTS_DIR = "results"

RESULTS_FILES = sorted(os.listdir(RESULTS_DIR))


def drop_bad_recalls(recalls_data_frame):
    """ Drop time steps with no memory recall """
    return recalls_data_frame.drop(
        recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
    )


recalls_data_frames = list(npy_loader_cont_forth.get_cont_forth_data_frames().values())

recalls_data_frames = [
    drop_bad_recalls(recall_data_frame) for recall_data_frame in recalls_data_frames
]


#%%


def get_memory_jumps_idx(recalls):
    """ When the recalled memory changes """

    # Get when there is a change in the recalled memory, a jump
    jumps = recalls.diff()
    # Eliminate all -1 from diff() to get ones after sum, where jumps happened
    jumps = jumps[jumps == 1]
    # Sum along columns to obtain the vector of jumps, 1 when jumped, 0 when not
    jumps_vector = jumps.sum(axis=1).values
    # Add its index information to the memory jump
    jumps_idx = jumps_vector * np.arange(jumps_vector.size)
    # Remove zeros to just return indices
    return jumps_idx.nonzero()[0]


#%%


def get_inter_retrieval_times(jumps_idx):
    """
    IRT is the number of iterations that pass before a new memory is recalled.
    e.g. p1, p1, p1, p5; IRT = 3
    """

    # There needs to be an imaginary jump just before first memory recall to compute IRTs
    first_jump_idx = [-1]
    # Stack an imaginary first jump to compute diffs correctly
    jumps_idx_probed = np.hstack((first_jump_idx, jumps_idx))
    # Diffs of jump indexes gives iterations until jump; these are the IRT
    return np.diff(jumps_idx_probed)


#%%

all_jumps = [
    get_memory_jumps_idx(recalls_data_frame)
    for recalls_data_frame in recalls_data_frames
]

all_irts = np.concatenate([get_inter_retrieval_times(jumps) for jumps in all_jumps])


#%%

axi = sns.distplot(all_irts, kde_kws={"bw": "0.5"})

axi.set_xlim(left=1, right=10)
axi.set_ylim(top=0.65)
axi.set(
    xlabel="Inter-retrieval times (cycles)",
    ylabel="Frequency",
    title="IRTs Distribution",
)
