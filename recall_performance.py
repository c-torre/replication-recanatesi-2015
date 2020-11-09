"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Utils for plotting network recalls analysis.
"""

import os
from typing import Hashable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm


def get_used_parameters(
    file_names: Hashable,
    sims_path: str,
    num_trials: int,
    separator="-",
) -> pd.DataFrame:
    """Gets and saves the set of parameters of the simulations"""

    def get_changing_parameter(file_name: str, position: int, separator="-") -> int:
        """Parse name to get parameters"""

        return file_name.split(separator)[position]

    parameters_used = pd.DataFrame(
        index=range(num_trials),
        columns=["seed", "cont_forth", "noise_var"],
    )

    parameters_used["seed"] = tuple(
        int(get_changing_parameter(file_name, position=0, separator=separator)[1:])
        for file_name in file_names
    )
    parameters_used["cont_forth"] = tuple(
        int(get_changing_parameter(file_name, position=1)[2:])
        for file_name in file_names
    )
    parameters_used["noise_var"] = tuple(
        int(get_changing_parameter(file_name, position=2)[1:-4])
        for file_name in file_names
    )

    parameters_used.to_csv(os.path.join(sims_path, "parameters.csv"))
    return parameters_used


def load_sequences(file_paths: Iterable) -> None:
    """Make recall sequences from saved ndarrays"""

    print("Loading recall sequences...")
    # recall_sequences = np.vstack([np.load(file_path) for file_path in file_paths])
    # loaded = [np.load(file_path) for file_path in file_paths]
    ## sequences_shapes = {}
    sequences_shapes = pd.DataFrame(columns=("Recall sequence", "Length"))
    num_files = len(file_paths)

    for file_idx, file_path in enumerate(file_paths):
        recall_sequence = np.load(file_path)
        new_row = {"Recall sequence": recall_sequence, "Length": len(recall_sequence)}
        sequences_shapes = sequences_shapes.append(new_row, ignore_index=True)

    ##intended_value = np.bincount(sequences_shapes.values()).argmax()
    ## recall_sequences = np.fromiter(
    ##     dict(
    ##         filter(lambda item: item[1] == intended_value, sequences_shapes.items())
    ##     ).keys()
    ## )
    ###filtered = sequences_shapes[sequences_shapes == sequences_shapes["Length"].mode()]
    #### intended_value = sequences_shapes["Length"].mode().to_numpy()

    # Take only arrays as long as the most common length
    intended_value = sequences_shapes["Length"].mode().array[0]
    filtered = sequences_shapes[sequences_shapes["Length"] == intended_value]
    recall_sequences = filtered["Recall sequence"].to_numpy()
    # print((filtered))
    # print(len(filtered))

    # loaded = [np.load(file_path) for file_path in file_paths]
    recall_sequences = np.vstack(recall_sequences)

    print("Done!")
    return recall_sequences  # THE GOOOOOD ONEEEEE


def make_similarity(
    similarity_paths: str,
    # num_trials: int,
) -> np.ndarray:
    """Get pattern similarities from saved ndarrays"""

    print("Loading pattern similarities...")
    similarities = np.stack(
        [np.load(similarity_path) for similarity_path in sorted(similarity_paths)]
    )
    print("Done!")
    return similarities


def get_main_df(
    num_trials: int,
    param_iterable: Iterable,
    recall_sequence: Iterable,
    pattern_similarities: Iterable,
) -> pd.DataFrame:
    """Make dataframe with all recall metrics"""

    print("Calculating recall metrics...")
    df = pd.DataFrame()
    for i_trial in tqdm(range(num_trials)):
        param = param_iterable[i_trial]
        items, times = np.unique(recall_sequence[i_trial], return_index=True)
        times = times[items != 0]
        times = np.sort(times)
        items = recall_sequence[i_trial][times]
        irts = np.insert(np.diff([times]), 0, 0)
        sizes = pattern_similarities[i_trial][items, items]
        ranksall = np.array(
            [np.argsort(x).argsort() for x in pattern_similarities[i_trial]]
        )
        intersections = np.insert(
            pattern_similarities[i_trial][items[:-1], items[1:]], 0, 0
        )  #
        ranks = np.insert(ranksall[items[:-1], items[1:]], 0, 0)
        dicttrial = {
            "items": items,
            "times": times,
            "trial": i_trial,
            "irts": irts,
            "sizes": sizes,
            "intersections": intersections,
            "ranks": ranks,
            "param": param,
        }
        dftrial = pd.DataFrame.from_dict(dicttrial)
        df = df.append(dftrial)

    df = df.reset_index().rename(columns={"index": "position"})
    N_recalled = (
        df.groupby("trial")
        .agg({"position": np.max})
        .rename(columns={"position": "N_recalled"})
    )
    df = df.merge(N_recalled, on="trial")
    df.N_recalled = df.N_recalled + 1  # This is to change that python is 0-based

    # Calculate for the varying parameter
    dft = df.groupby("param").apply(np.mean)
    print("Done!")

    return dft
