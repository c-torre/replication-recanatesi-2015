import os

import numpy as np
import pandas as pd

PLOTS_DIR = "plt"
RESULTS_DIR = "results"
CONT_FORTH_DIR = os.path.join(RESULTS_DIR, "forward-contiguity")

CONT_FORTH_FILES = sorted(os.listdir(CONT_FORTH_DIR))


def get_cont_forth_name_values():
    # List each part of the file names separated by "-", get second element and remove "j"
    # This gets forward contiguity parameter value from file names
    return [
        int(file_name.split("-")[1][1:])
        for file_name in CONT_FORTH_FILES
        if "j" in file_name
    ]


def get_recall_arrays():

    return [
        np.load(os.path.join(CONT_FORTH_DIR, file_name))
        for file_name in CONT_FORTH_FILES
        if "j" in file_name
    ]


def get_cont_forth_data_frames():

    cont_forths = get_cont_forth_name_values()

    recall_arrays_contiguity = get_recall_arrays()

    recall_memory_indices = [
        f"memory_{memory_num}"
        for memory_num in range(recall_arrays_contiguity[0].shape[0])
    ]

    recall_data_frames_contiguity = {
        cont_forths[idx]: pd.DataFrame(
            data=np.rot90(recall_data_array), columns=recall_memory_indices
        )
        for idx, recall_data_array in enumerate(recall_arrays_contiguity)
    }

    return recall_data_frames_contiguity


if __name__ == "__main__":
    cont_forth_data_frames_dict = get_cont_forth_data_frames()
