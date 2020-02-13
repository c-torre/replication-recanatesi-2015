#%%
import pandas as pd
import seaborn as sns

import recall_performance.npy_loader_cont_forth

cont_forth_data_frames_dict = (
    recall_performance.npy_loader_cont_forth.get_cont_forth_data_frames()
)

#%%

def drop_bad_recalls(recalls_data_frame):
    """ Drop time steps with no memory recall """
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


data = get_recall_performance_cont_forth(cont_forth_data_frames_dict)


def plot_recalled_memories_cont_forth(data):
    ax = sns.scatterplot(data=data)
    ax.set(
        xlabel="Forward contiguity $J_{forth}$",
        ylabel="Number of recalled memories",
        title="Recalled Memories per $J_{forth}$",
    )


if __name__ == "__main__":
    plot_recalled_memories_cont_forth(data)

# #%% Cummulative recalls (just when a memory changes)
# # Get totall different memory recalls by:
# # - Checking the difference of recall against the previous iteration
# # - If there is a memory jump, diff is 1 for the new memory and -1 for previous
# # - Sum along both axes to obtain the cummulative recalls

# spam = test_data[test_data.diff() == 1].sum().sum()

