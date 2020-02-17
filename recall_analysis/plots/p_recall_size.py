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
concatting = combined_recalls_intersections_all[0]
for idx in range(len(combined_recalls_intersections_all)):
    if idx == 0:
        continue
    concatting = concatting.append(combined_recalls_intersections_all[idx])


a = concatting["transition_size"].value_counts()
a /= a.sum()
#%%
def plot_cumulative_recalled_memories(data):
    ax = sns.lineplot(data=a)
    ax.set(
        xlabel="",
        ylabel="Number of appereances",
        title="IRTs Distribution",
    )


if __name__ == "__main__":
    plot_cumulative_recalled_memories(plot_cumulative_recalled_memories)

