#%%
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import recall_analysis.data_processing.main
import paths

file_pkl = os.path.join(paths.BKP_DIR, "combined_recalls_intersections.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

combined_recalls_intersections_all = pickle.load(open(file_pkl, "rb"))


#%%
# Put all irts together for all df and count how many times appear each irt
to_split = combined_recalls_intersections_all[0]
for idx in range(len(combined_recalls_intersections_all)):
    if idx == 0:
        continue
    to_split = to_split.append(combined_recalls_intersections_all[idx])

to_split = to_split["transition_size"]
to_split = to_split.sort_values()
to_split = to_split[to_split != 0].values
#%%
# Make Windows
max_ = np.amax(to_split)
min_ = np.amin(to_split)
step = (max_ - min_) / 15
rank_values = np.arange(start=min_, stop=max_, step=step)
rank_dict = {}
#%%
# How many fit per bin
a = np.digitize(to_split, rank_values)
a = pd.Series(a)
a = a.value_counts()
# Do proportion
a /= a.sum()
#%%
def plot_cumulative_recalled_memories(data):
    ax = sns.lineplot(data=a)
    ax.set(
        xlabel="Rank by transition intersection size",
        ylabel="Proportion of memory transitions",
        title="Distribution of Transitions by Intersection Size",
    )


if __name__ == "__main__":
    plot_cumulative_recalled_memories(plot_cumulative_recalled_memories)
