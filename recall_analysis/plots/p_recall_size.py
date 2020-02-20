# #%%
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import seaborn as sns

# import recall_analysis.data_processing.main
# import paths

# import paths
# from recall_analysis.data_processing.pre_processing import loader_utils

# # POPULATIONS_MEMORIES_DIR = paths.POPULATIONS_MEMORIES_DIR
# # POPULATION_SIZES_DIR = paths.POPULATION_SIZES_DIR
# # file_pkl = os.path.join(paths.BKP_DIR, "combined_recalls_intersections.p")

# # populations_memories = loader_utils.get_arrays_from_files(POPULATIONS_MEMORIES_DIR)
# # populations_memories = loader_utils.arrays_to_data_frames(populations_memories)
# # population_sizes = loader_utils.get_arrays_from_files(POPULATION_SIZES_DIR)

# # if not os.path.exists(file_pkl):
# #     recall_analysis.data_processing.main.make_pickles()

# combined_recalls_intersections_all = pickle.load(open(file_pkl, "rb"))
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



#%%

sizes_series = [pd.Series(array, name="memory_size") for array in population_sizes]


#%%

spam = pd.merge(
    example_combined, example_size, left_on="memory_recalled", right_index=True
)


#%%

mergeds = [
    pd.merge(combined, size, left_on="memory_recalled", right_index=True)
    for combined, size in zip(combined_recalls_intersections_all, sizes_series)
]


#%%

concatting = mergeds[0]

for idx in range(len(mergeds)):
    if idx == 0:
        continue
    concatting = concatting.append(mergeds[idx])


#%%
a = concatting["transition_size"].value_counts()
#%%
a /= a.sum()
#%%
def plot_cumulative_recalled_memories(data):
    ax = sns.lineplot(data=a)
    ax.set(
        xlabel="Memory size (neurons)",
        ylabel="Proportion of recalls",
        title="Proportion of Recalls with Memory Size",
    )


if __name__ == "__main__":
    plot_cumulative_recalled_memories(plot_cumulative_recalled_memories)

