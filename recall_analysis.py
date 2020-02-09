#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import paths

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


reca = get_recall_performance_cont_forth(recalls_data_frames_su)

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
