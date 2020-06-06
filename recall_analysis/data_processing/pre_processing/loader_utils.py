#%%
"""

"""

import os

import numpy as np
import pandas as pd


def identify_results_type(file_name):
    # Warning may return invalid param in intersections (corrected later for it)
    splits = file_name.split("-")
    # First file name element is the seed
    seed = splits[0]
    # Second element contains modified parameter, if any
    param_indicator = splits[1]
    if param_indicator[0] == "j" or param_indicator[0] == "n":
        return int(param_indicator[1:])
    return seed


def get_arrays_from_files(results_type_dir, calling_intersections=False):
    """ Loads all .npy files into a list from the indicated dir """

    files = sorted(os.listdir(results_type_dir))
    assert files
    array_dicts = {
        identify_results_type(file_name): np.load(
            os.path.join(results_type_dir, file_name)
        )
        for file_name in files
        if ".npy" in file_name
    }

    if calling_intersections:
        return list(array_dicts.values())
    return array_dicts


def _get_recalls_data_frames(recalls_arrays):
    """ Array to data frame """

    return {
        mod_param: pd.DataFrame(
            np.rot90(recalls_array),
            columns=[
                f"memory{memory_idx}" for memory_idx in range(recalls_array.shape[0])
            ],
        )
        for mod_param, recalls_array in recalls_arrays.items()
    }


def _drop_bad_recalls(recalls_data_frames):
    """ Drop time steps with no memory recall """

    return {
        mod_param: (
            recalls_data_frame.drop(
                recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
            )
        )
        for mod_param, recalls_data_frame in recalls_data_frames.items()
    }


def get_cleaned_frames(results_type_dir):
    """ From results files to cleaned data frames """

    recalls_arrays_dict = get_arrays_from_files(results_type_dir)
    recalls_data_frames_dicts = _get_recalls_data_frames(recalls_arrays_dict)
    return _drop_bad_recalls(recalls_data_frames_dicts)


def arrays_to_data_frames(arrays):
    """ Array to data frame """

    return [pd.DataFrame(array) for array in arrays]


#%%
if __name__ == "__main__":
    import paths

    SEED_RESULTS_DIR = paths.SEED_RESULTS_DIR
    NOISE_RESULTS_DIR = paths.NOISE_RESULTS_DIR
    CONT_RESULTS_DIR = paths.CONT_RESULTS_DIR
    populations_memories_dir = paths.POPULATIONS_MEMORIES_DIR
    population_sizes_dir = paths.POPULATION_SIZES_DIR
    files_path = populations_memories_dir
    spam = get_cleaned_frames(files_path)


# %%
