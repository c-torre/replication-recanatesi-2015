import numpy as np
import pandas as pd
import seaborn as sns

import recall_analysis.npy_loader_cont_forth

cont_forth_data_frames_dict = (
    recall_performance.npy_loader_cont_forth.get_cont_forth_data_frames()
)


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


cumulative_recalls = get_cumulative_recalls(cont_forth_data_frames_dict)

#%%


def plot_cumulative_recalled_memories(data):
    ax = sns.lineplot(data=data)
    ax.set(
        xlabel="Time (cycles)",
        ylabel="Average cumulative number of recalled memories",
        title="Cumulative Memory Recalls",
    )


if __name__ == "__main__":
    plot_cumulative_recalled_memories(cumulative_recalls)
