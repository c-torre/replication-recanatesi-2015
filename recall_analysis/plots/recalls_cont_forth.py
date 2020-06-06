#%%
# import pandas as pd
# import seaborn as sns

# import recall_analysis.npy_loader_cont_forth

# cont_forth_data_frames_dict = (
#     recall_performance.npy_loader_cont_forth.get_cont_forth_data_frames()
# )
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import recall_analysis.data_processing.main
import paths

file_pkl = os.path.join(paths.BKP_DIR, "recalls_frames_cont.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

recalls_analysis_data_frames_all = pickle.load(open(file_pkl, "rb"))

#%%


# def drop_bad_recalls(recalls_data_frame):
#     """ Drop time steps with no memory recall """
#     return recalls_data_frame.drop(
#         recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
#     )


# def get_recalled_memories(recalls_data_frame):
#     # Get number of memories that were ever recalled
#     return sum(recalls_data_frame.sum() > 0)


# def get_recall_performance_cont_forth(recalls_data_frames):
#     for cont_forth, recalls_data_frame in recalls_data_frames.items():
#         recalls_data_frames[cont_forth] = drop_bad_recalls(recalls_data_frame)
#         recalls_data_frames[cont_forth] = get_recalled_memories(recalls_data_frame)

#     return pd.Series(recalls_data_frames, name="recall_for_con")
data = [
    recalls_analysis_data_frame.iloc[-1]
    for recalls_analysis_data_frame in recalls_analysis_data_frames_all
]
#%%
data_plot = [data_series.loc[["mod_param", "unique_recalls_cum_sum"]] for data_series in data]


#%%

data_good = pd.DataFrame(data_plot)
# data = get_recall_performance_cont_forth(cont_forth_data_frames_dict)


data_int = data_good["mod_param"].values
iterator = np.nditer(data_int, flags=["refs_ok"], op_flags=["readwrite"])

with iterator:
    for i in iterator:
        i[...] = int(str(i)[1:])

data_good["mod_param"] = data_int
#%%


def plot_recalled_memories_cont_forth(data):
    ax = sns.scatterplot(x="mod_param", y="unique_recalls_cum_sum", data=data)
    ax.set(
        title="Recalled Memories per Forward Contiguity Value",
        xlabel="$\kappa_{forth}$",
        ylabel="Number of unique memories recalled",
    )


if __name__ == "__main__":
    plot_recalled_memories_cont_forth(data_good)

# #%% Cummulative recalls (just when a memory changes)
# # Get totall different memory recalls by:
# # - Checking the difference of recall against the previous iteration
# # - If there is a memory jump, diff is 1 for the new memory and -1 for previous
# # - Sum along both axes to obtain the cummulative recalls

# spam = test_data[test_data.diff() == 1].sum().sum()



# %%
