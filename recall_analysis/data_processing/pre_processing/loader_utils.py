import os

import numpy as np
import pandas as pd


def get_arrays_from_files(results_type_dir):
    """ Loads all .npy files into a list from the indicated dir """

    files = sorted(os.listdir(results_type_dir))
    assert files
    return [
        np.load(os.path.join(results_type_dir, file_name))
        for file_name in files
        if ".npy" in file_name
    ]


def _get_recalls_data_frames(recalls_arrays):
    """ Array to data frame """

    return [
        pd.DataFrame(
            np.rot90(recalls_array),
            columns=[
                f"memory{memory_idx}" for memory_idx in range(recalls_array.shape[0])
            ],
        )
        for recalls_array in recalls_arrays
    ]


def _drop_bad_recalls(recalls_data_frames):
    """ Drop time steps with no memory recall """

    return [
        (
            recalls_data_frame.drop(
                recalls_data_frame[recalls_data_frame.sum(axis=1) == 0].index
            )
        )
        for recalls_data_frame in recalls_data_frames
    ]


def get_cleaned_frames(results_type_dir):
    """ From results files to cleaned data frames """

    recalls_arrays = get_arrays_from_files(results_type_dir)
    recalls_data_frames = _get_recalls_data_frames(recalls_arrays)
    return _drop_bad_recalls(recalls_data_frames)


def arrays_to_data_frames(arrays):
    """ Array to data frame """

    return [pd.DataFrame(array) for array in arrays]
