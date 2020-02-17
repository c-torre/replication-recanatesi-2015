"""

"""

#%%

import numpy as np
import pandas as pd
from tqdm import tqdm

import paths
from recall_analysis.data_processing.pre_processing import loader_utils

#%%


def make_master_data_frame(recalls_data_frame):
    """ Create the data frame that contains most information for plots """

    return pd.DataFrame(index=range(len(recalls_data_frame.index)))


# recall_analysis_data_frame = make_master_data_frame()


def _get_memory_jumps_idx(recalls):
    """ When the recalled memory changes """

    # Get when there is a change in the recalled memory, a jump
    jumps = recalls.diff()
    # Eliminate all -1 from diff() to get ones after sum, where jumps happened
    jumps = jumps[jumps == 1]
    # Sum along columns to obtain the vector of jumps, 1 when jumped, 0 when not
    jumps_vector = jumps.sum(axis=1).values
    # Add its index information to the memory jump
    jumps_idx = jumps_vector * np.arange(jumps_vector.size)
    # Remove zeros to just return indices
    return jumps_idx.nonzero()[0]


# example_jumps = get_memory_jumps_idx(example_frame)


def _get_inter_retrieval_times(jumps_idx):
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


# example_irts = get_inter_retrieval_times(example_jumps)


def _expand_irts(irts, recalls_data_frame):
    """
    Transform a group of IRTs into its time series.
    e.g. (3, 2, 1) to (0, 0, 3, 0, 2, 1)
    """

    if irts.size != 0:
        return np.hstack([(np.hstack((np.zeros(irt - 1), irt))) for irt in irts])
    else:
        return np.zeros(recalls_data_frame.shape[0])


def get_irts_time_series(recalls):
    """ Get time indexed inter-retrieval times """

    jumps_idx = _get_memory_jumps_idx(recalls)
    inter_retrieval_times = _get_inter_retrieval_times(jumps_idx)
    return pd.Series(_expand_irts(inter_retrieval_times, recalls), name="irt")


#%%


def get_vectorized_probed_recalls(recalls_data_frame):
    # Duplicate?

    probe = np.cumsum(np.ones_like(recalls_data_frame), axis=1) - 1
    probed_recalls = recalls_data_frame * (probe)
    vectorized_probed_recalls = probed_recalls.sum(axis=1)
    return vectorized_probed_recalls


#%%


def get_jumps(vectorized_recalls):
    # Duplicate?

    jumps_absolute = vectorized_recalls.diff().dropna()
    jumps_bool = (jumps_absolute != 0).astype(int)
    return pd.Series(jumps_bool, name="transition", dtype=bool)


#%% Hunting the first appereance


def get_first_unique_recalls(vectorized_probed_recalls, recalls_analysis_data_frame):
    # Get which unique memories were recalled
    unique_memory_recalls = vectorized_probed_recalls.unique()

    # Get time indices of the unique recalls
    unique_memory_recalls_idx = sorted(
        [
            vectorized_probed_recalls.eq(unique_memory_recall).idxmax()
            for unique_memory_recall in unique_memory_recalls
        ]
    )

    # Set an array with ones in the indices of unique recall
    first_appereances = pd.Series(np.zeros(recalls_analysis_data_frame.shape[0]))
    first_appereances.iloc[unique_memory_recalls_idx] = 1
    return pd.Series(first_appereances, name="unique_recalls")


def get_counts_unique_recalls(first_appereances):
    # Get counts from first appereances in previous array
    counter = 1
    counts_first_appereances = np.zeros_like(first_appereances)  # X
    for idx, element in enumerate(first_appereances):
        if element == 1:
            counts_first_appereances[idx] = counter
            counter += 1

    return pd.Series(counts_first_appereances, name="unique_recall_count")


#%% Loop all


def make_all(files_path):

    recalls_data_frames = loader_utils.get_cleaned_frames(files_path)
    # Avoid division by zero exception later with hacky sum
    epsilon = 0.000000001

    all_data = []

    print("Building all data frames for recalls analysis...")
    for recalls_data_frame in tqdm(recalls_data_frames):

        # Make the data frame that will contain all info indexed by time cycle
        recalls_analysis_data_frame = make_master_data_frame(recalls_data_frame)

        # IRT
        recalls_analysis_data_frame["irt"] = get_irts_time_series(recalls_data_frame)
        # IRT cumulative sum
        recalls_analysis_data_frame["irts_cum_sum"] = recalls_analysis_data_frame[
            "irt"
        ].cumsum()
        # Memory recalled
        recalls_analysis_data_frame["memory_recalled"] = get_vectorized_probed_recalls(
            recalls_data_frame
        )
        # Transition (bool), if happened
        recalls_analysis_data_frame["transition"] = get_jumps(
            recalls_analysis_data_frame["memory_recalled"]
        )
        # Cumulative sum of the transitions
        recalls_analysis_data_frame[
            "transitions_cum_sum"
        ] = recalls_analysis_data_frame["transition"].cumsum()
        # Average IRTs dividing cumulative IRTs by cumulative transitions
        recalls_analysis_data_frame["average_irts"] = (
            recalls_analysis_data_frame["irts_cum_sum"] + epsilon
        ) / (recalls_analysis_data_frame["transitions_cum_sum"] + epsilon)
        # New memory is recalled which was not recalled before, add 1 at time cycle it happens, else 0
        unique_recalls = get_first_unique_recalls(
            recalls_analysis_data_frame["memory_recalled"], recalls_analysis_data_frame
        )
        recalls_analysis_data_frame["unique_recalls_cum_sum"] = unique_recalls.cumsum()
        recalls_analysis_data_frame["unique_recall_count"] = get_counts_unique_recalls(
            unique_recalls
        )
        all_data.append(recalls_analysis_data_frame)

    return all_data


#%% Go

# if __name__ == "__main__":
#     recalls_analysis_data_frames = make_all()
