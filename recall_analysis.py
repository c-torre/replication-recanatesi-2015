#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PLOTS_DIR = "plt"
RESULTS_DIR = "results"

RESULTS_FILES = sorted(os.listdir(RESULTS_DIR))


def get_cont_forth_name_values():
    # List each part of the file names separated by "-", get second element and remove "j"
    # This gets forward contiguity parameter value from file names
    return [
        int(file_name.split("-")[1][1:])
        for file_name in RESULTS_FILES
        if "j" in file_name
    ]


def get_recall_arrays():

    return [
        np.load(os.path.join(RESULTS_DIR, file_name))
        for file_name in RESULTS_FILES
        if "j" in file_name
    ]


#%%


def get_cont_forth_data_frames():

    cont_forths = get_cont_forth_name_values()

    recall_arrays_contiguity = get_recall_arrays()

    recall_memory_indices = [
        f"memory_{memory_num}"
        for memory_num in range(recall_arrays_contiguity[0].shape[0])
    ]

    recall_data_frames_contiguity = {
        cont_forths[idx]: pd.DataFrame(
            data=np.rot90(recall_data_array), columns=recall_memory_indices
        )
        for idx, recall_data_array in enumerate(recall_arrays_contiguity)
    }

    return recall_data_frames_contiguity


recalls_data_frames_su = get_cont_forth_data_frames()


#%%


def drop_bad_recalls(recalls_data_frame):
    # Get number of memories that were ever recalled
    return recalls_data_frame.drop(
        recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
    )


def get_recalled_memories(recalls_data_frame):
    # Get number of memories that were ever recalled
    return sum(recalls_data_frame.sum() > 0)


def get_recall_performance_cont_forth(recalls_data_frames):
    for cont_forth, recalls_data_frame in recalls_data_frames.items():
        recalls_data_frames[cont_forth] = drop_bad_recalls(recalls_data_frame)
        recalls_data_frames[cont_forth] = get_recalled_memories(recalls_data_frame)

    return pd.Series(recalls_data_frames, name="recall_for_con")


reca = get_recall_performance_cont_forth(recalls_data_frames_su.copy())


#%% Plot
ax = sns.scatterplot(data=reca)
ax.set(
    xlabel="Time (cycles)",
    ylabel="Number of recalled memories",
    title="Recalled Memories per $J_{forth}$",
)


# #%% Cummulative recalls (just when a memory changes)
# # Get totall different memory recalls by:
# # - Checking the difference of recall against the previous iteration
# # - If there is a memory jump, diff is 1 for the new memory and -1 for previous
# # - Sum along both axes to obtain the cummulative recalls

# spam = test_data[test_data.diff() == 1].sum().sum()

#%%


def get_cumulative_recalls(recalls_data_frames):
    """ Build an array with average cumulative recall of the first recall of recalled memories """

    cumulative_vectors = []
    for recall_data_frame in recalls_data_frames.values():
        # Get first time index a memory is recalled, taking care of not recalled
        memory_first_recall_idx = pd.unique(recall_data_frame.idxmax())
        # Pre-allocate vector
        cumulative_vector = np.zeros(recall_data_frame.shape[0])
        # Add 1 to all elements from the first recall until the end of array
        for memory_recall_idx in memory_first_recall_idx:
            cumulative_vector[memory_recall_idx:] += 1
        cumulative_vectors.append(cumulative_vector)

    return np.average(np.vstack(cumulative_vectors), axis=0)


cumulative_recalls = get_cumulative_recalls(recalls_data_frames_su)

#%%

ax = sns.lineplot(data=cumulative_recalls)
ax.set(
    xlabel="Time (cycles)",
    ylabel="Average cumulative number of recalled memories",
    title="Cumulative Memory Recalls",
)

#%%

recalls_data_frames = recalls_data_frames_su[600]


def get_memory_jumps_idx(recalls):
    """ When the recalled memory changes """

    recalls = recalls_data_frames
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


jumpos = get_memory_jumps_idx(recalls_data_frames)

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


irts = get_inter_retrieval_times(jumpos)

