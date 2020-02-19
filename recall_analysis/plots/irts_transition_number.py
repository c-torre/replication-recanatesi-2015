#%%
# #%%
# import os

# import numpy as np
# import pandas as pd
# import seaborn as sns

# from recall_analysis import npy_loader_cont_forth

# PLOTS_DIR = "plt"
# RESULTS_DIR = "results"

# RESULTS_FILES = sorted(os.listdir(RESULTS_DIR))


# def drop_bad_recalls(recalls_data_frame):
#     """ Drop time steps with no memory recall """
#     return recalls_data_frame.drop(
#         recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
#     )


# recalls_data_frames = list(npy_loader_cont_forth.get_cont_forth_data_frames().values())

# recalls_data_frames = [
#     drop_bad_recalls(recall_data_frame) for recall_data_frame in recalls_data_frames
# ]
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

import paths
import recall_analysis.data_processing.main

file_pkl = os.path.join(paths.BKP_DIR, "recalls_frames_seeds.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

recalls_analysis_data_frames_all = pickle.load(open(file_pkl, "rb"))

#%%


# spam = recalls_analysis_data_frames_all[0]["unique_recalls_cum_sum"].iloc[-1]


# branches = np.unique(
#     [
#         recalls_analysis_data_frame["unique_recalls_cum_sum"].iloc[-1]
#         for recalls_analysis_data_frame in recalls_analysis_data_frames_all
#     ]
# )

indexed = [
    pd.Series(
        recalls_analysis_data_frame["unique_irt"],
        name=recalls_analysis_data_frame["unique_recalls_cum_sum"].iloc[-1],
    )
    for recalls_analysis_data_frame in recalls_analysis_data_frames_all
]

#%%
indexed = [
    pd.Series(
        series.values[series.values.nonzero()[0]],
        name=series.name
    )
    for series in indexed
]
#%%
# indexed = [series.reset_index(drop=True) for series in indexed]

#%%
# indexed_reduced = [
#     series[series.nonzero()[0]] for series in indexed
# ]

#%%

spam = pd.DataFrame(indexed)


# ["transitions_cum_sum",
# "average_irts"  # y
# ]
#%%

eggs = spam.groupby(by=spam.index).sum().reset_index(drop=True)
#%%
eggs /= len(recalls_analysis_data_frames_all)
eggs.index += 1
eggs.index.name = "average_irt_unique_transition"
#%%
eggerino = eggs.dropna(axis=1, how="all")  # change me to replace by 0s
eggerino = eggerino.loc[:, (eggerino != 0).any(axis=0)]  # Drop column only 0
#%%
# pd.concat([eggs], keys=['foo'], names=['Firstlevel'])

#%%

example_frame = recalls_analysis_data_frames_all[5]

branches_plot_data = pd.DataFrame(
    index=range(len(example_frame.index)),
    columns=["new_recall_jumps", "jumps_cum_sum", "irts_cum_sum"],
)


def get_memory_jumps_idx(recalls):
    """ When the recalled memory changes """

    # Get when there is a change in the recalled memory, a jump
    jumps = recalls.diff()
    # Eliminate all -1 from diff() to get ones after sum, where jumps happened
    jumps = jumps[jumps == 1]
    # Sum along columns to obtain the vector of jumps, 1 when jumped, 0 when not
    jumps_vector = jumps.sum(axis=1)
    # Add its index information to the memory jump
    jumps_idx = jumps_vector * np.arange(jumps_vector.size)
    # Remove zeros to just return indices
    return jumps_idx.nonzero()[0]


example_jumps = get_memory_jumps_idx(example_frame)

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


example_irts = get_inter_retrieval_times(example_jumps)
#%% Building the array for the multi gradient line plot


def expand_irts(irts):
    """
    Transform a group of IRTs into its time series.
    e.g. (3, 2, 1) to (0, 0, 3, 0, 2, 1)
    """

    return np.hstack([(np.hstack((np.zeros(irt - 1), irt))) for irt in irts])


irts_time_series = expand_irts(example_irts)  # Y num
irts_cum_sum = pd.Series(irts_time_series.cumsum())
branches_plot_data.loc[:, "irts_cum_sum"] = irts_cum_sum

#%%

probe = np.cumsum(np.ones_like(example_frame), axis=1) - 1
probed_recalls = example_frame * (probe)
vectorized_probe_recalls = probed_recalls.sum(axis=1)
jumps_absolute = vectorized_probe_recalls.diff().dropna()
jumps_bool = (jumps_absolute != 0).astype(int)
jumps_cumsum = jumps_bool.cumsum()
jumps_cum_sum_time_series = pd.concat((pd.Series([0]), jumps_cumsum))  # Y den
branches_plot_data.loc[:, "jumps_cum_sum"] = jumps_cum_sum_time_series
#%% Hunting the first appereance


unique_memory_recalls = vectorized_probe_recalls.unique()
#%%
unique_memory_recalls_idx = sorted(
    [
        vectorized_probe_recalls.eq(unique_memory_recall).idxmax()
        for unique_memory_recall in unique_memory_recalls
    ]
)

#%%
first_appereances = pd.Series(np.zeros_like(vectorized_probe_recalls))
#%%
first_appereances.iloc[unique_memory_recalls_idx] = 1

counter = 1
counts_first_appereances = np.zeros_like(first_appereances)  # X
for idx, element in enumerate(first_appereances):
    if element == 1:
        counts_first_appereances[idx] = counter
        counter += 1


branches_plot_data.loc[:, "new_recall_jumps"] = counts_first_appereances

# %%

branches_plot_data["average_irts"] = (
    branches_plot_data["irts_cum_sum"] / branches_plot_data["jumps_cum_sum"]
)

#%%

a = pd.melt(eggs,)

#%%

counts_distribution = recalls_analysis_data_frames_all[0]
for idx in range(len(recalls_analysis_data_frames_all)):
    if idx == 0:
        continue
    counts_distribution = counts_distribution.append(
        recalls_analysis_data_frames_all[idx]
    )

#%%
counts_series = counts_distribution["irt"]
counts_series = counts_series[counts_series != 0].value_counts()


#%%

data = eggs.loc[1]
non_zeros = eggs.loc[1].nonzero()[0]
data = data[non_zeros]

#%%

for num_unique_recalls in list(eggerino.index):
    data = eggerino.loc[num_unique_recalls]
    # non_zeros = eggs.loc[num_unique_recalls].nonzero()[0]
    # data = data[non_zeros]
    # # data.index = np.arange(num_unique_recalls + 1)
    # data = data.reset_index(drop=True)   # print(data)
    # print(data)
    sns.lineplot(data=data)
# sns.lineplot(x="new_recall_jumps", y="average_irts", data=branches_plot_data)
# sns.lineplot(data=)



# %%
