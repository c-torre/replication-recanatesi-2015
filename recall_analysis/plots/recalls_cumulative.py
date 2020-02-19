#%%
import os
import pickle

import numpy as np
import seaborn as sns

import paths
import recall_analysis.data_processing.main

file_pkl = os.path.join(paths.BKP_DIR, "recalls_frames_seeds.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

recalls_analysis_data_frames_all = pickle.load(open(file_pkl, "rb"))

# import recall_analysis.npy_loader_cont_forth

# cont_forth_data_frames_dict = (
#     recall_performance.npy_loader_cont_forth.get_cont_forth_data_frames()
# )


# def get_cumulative_recalls(recalls_data_frames):
#     """ Build an array with average cumulative recall of the first recall of recalled memories """

#     cumulative_vectors = []
#     for recall_data_frame in recalls_data_frames.values():
#         # Get first time index a memory is recalled, taking care of not recalled
#         memory_first_recall_idx = pd.unique(recall_data_frame.idxmax())
#         # Pre-allocate vector
#         cumulative_vector = np.zeros(recall_data_frame.shape[0])
#         # Add 1 to all elements from the first recall until the end of array
#         for memory_recall_idx in memory_first_recall_idx:
#             cumulative_vector[memory_recall_idx:] += 1
#         cumulative_vectors.append(cumulative_vector)

#     return np.average(np.vstack(cumulative_vectors), axis=0)


# cumulative_recalls = get_cumulative_recalls(cont_forth_data_frames_dict)

#%%

cumulative_averages = np.average(
    [
        recalls_analysis_data_frame["unique_recalls_cum_sum"]
        for recalls_analysis_data_frame in recalls_analysis_data_frames_all
    ]
)

#%%


def plot_cumulative_recalled_memories(data):
    ax = sns.lineplot(data=cumulative_averages)
    ax.set(
        xlabel="Time (cycles)",
        ylabel="Average cumulative recalled memories",
        title="Cumulative Memory Recalls",
    )


if __name__ == "__main__":
    plot_cumulative_recalled_memories(plot_cumulative_recalled_memories)
