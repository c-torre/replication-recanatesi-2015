#%%
from recall_analysis import data_build, memory_intersections
import numpy as np
import pandas as pd

recall_analysis_data_frame = data_build.recall_analysis_data_frame
intersection_sizes = memory_intersections.intersection_sizes


#%%
after_jumps = recall_analysis_data_frame["memory_recalled"][
    recall_analysis_data_frame["transition"] == 1
]
after_jumps_idx = after_jumps.index

before_jumps_idx = after_jumps_idx - 1
before_jumps = recall_analysis_data_frame.loc[before_jumps_idx, "memory_recalled"]
#%%
transition_memories = pd.Series(
    [sorted(transition) for transition in zip(before_jumps, after_jumps)],
    name="transition_memories",
)

transition_sizes = pd.Series(
    [intersection_sizes.loc[tuple(size)] for size in transition_memories]
)

#%%

transition_memories.index = transition_sizes.index = after_jumps.index


#%%

recall_analysis_data_frame["transition_memories"] = transition_memories
recall_analysis_data_frame["transition_size"] = transition_sizes

# %%
recall_analysis_data_frame = recall_analysis_data_frame.dropna(0)
recall_analysis_data_frame["transition_memories"][
    recall_analysis_data_frame["transition_memories"] != 0
]


#%%
arr = np.array(sorted(recall_analysis_data_frame["transition_size"]))

#%% Make groups

possible_group_boudaries = np.unique(arr)
#%%
grouped = np.split(arr, 10)
