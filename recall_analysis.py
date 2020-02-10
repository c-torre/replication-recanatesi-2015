#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PLOTS_DIR = "plt"
RESULTS_DIR = "results"


def get_cont_forth_name_values():
    # List each part of the file names separated by "-", get second element and remove "j"
    # This gets forward contiguity parameter value from file names
    return [
        int(file_name.split("-")[1][1:])
        for file_name in sorted(os.listdir(RESULTS_DIR))
    ]


def get_recall_arrays():

    return [
        np.load(os.path.join(RESULTS_DIR, file_name))
        for file_name in sorted(os.listdir(RESULTS_DIR))
        if "j" and ".npy" in file_name
    ]


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


def get_cumulative_recalls(recalls_data_frames)
""" Build an array with average cumulative recall of the first recall of every recalled memory """

    cumulative_vectors = []
    for recall_data_frame in recall_data_frames.values():
        # Get first time index a memory is recalled, taking care of not recalled
        memory_first_recall_idx = pd.unique(recall_data_frame.idxmax())
        cumulative_vector = np.zeros(recall_data_frame.shape[0])
        # Add 1 to all elements from the first recall until the end of array
        for memory_recall_idx in memory_first_recall_idx:
            cumulative_vector[memory_recall_idx:] += 1
        cumulative_vectors.append(cummulative_vector)
    
    return np.average(np.vstack(cummulative_vectors), axis=0)

cumulative_recalls = get_cumulative_recalls(recall_data_frames_su)

#%%

ax = sns.lineplot(data=cummulative_recalls)
ax.set(
    xlabel="Time (cycles)",
    ylabel="Average cumulative number of recalled memories",
    title="Cumulative Memory Recalls",
)

#%%

recall_data_frames = recalls_data_frames_su[600]




def get_memory_jumps(recalls):
    """ When the recalled memory changes """

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
    # Diffs of jump indexes gives iterations until jump; these are the IRT
    inter_retrieval_times = np.diff(where_jumps_probed)

    return inter_retrieval_times

