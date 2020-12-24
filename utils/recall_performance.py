"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Utils for plotting network recalls analysis.
"""

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_used_parameters(
    file_names: Iterable,
    sims_path: str,
    num_trials: int,
    separator="-",
) -> pd.DataFrame:
    """Gets and saves the set of parameters of the simulations"""

    def get_changing_parameter(file_name: str, position: int, separator="-") -> int:
        """Parse file name to get parameters"""

        root, _ = os.path.splitext(file_name)
        selected_param = root.split(separator)[position]
        match = re.search("[0-9]+$", selected_param)  # Match any last number(s)

        return int(match.group())

    changing_parameters = "seed", "cont_forth", "noise_var"

    parameters_used = pd.DataFrame(
        index=range(num_trials),
        columns=changing_parameters,
    )

    for idx, parameter_name in enumerate(changing_parameters):
        parameters_used[parameter_name] = tuple(
            (get_changing_parameter(file_name, position=idx, separator=separator))
            for file_name in file_names
        )

    parameters_used.to_csv(os.path.join(sims_path, "parameters.csv"))
    return parameters_used


def load_sequences(file_paths: Iterable) -> None:
    """Make recall sequences from saved ndarrays"""

    print("Loading recall sequences...")
    sequences_shapes = pd.DataFrame(columns=("Recall sequence", "Length"))

    for file_path in tqdm(file_paths):
        recall_sequence = np.load(file_path)
        new_row = {"Recall sequence": recall_sequence, "Length": len(recall_sequence)}
        sequences_shapes = sequences_shapes.append(new_row, ignore_index=True)

    # Take only arrays as long as the most common length
    intended_value = sequences_shapes["Length"].mode().array[0]
    filtered = sequences_shapes[sequences_shapes["Length"] == intended_value]
    recall_sequences = filtered["Recall sequence"].to_numpy()
    recall_sequences = np.vstack(recall_sequences)

    print("Done!")
    return recall_sequences


def make_similarity(
    similarity_paths: Iterable,
) -> np.ndarray:
    """Get pattern similarities from saved ndarrays"""

    print("Loading pattern similarities...")
    similarities = []
    for similarity_path in tqdm(sorted(similarity_paths)):
        similarities.append(np.load(similarity_path))

    print("Done!")
    return np.stack(similarities)


def get_main_df(
    num_trials: int,
    param_iterable: Iterable,
    recall_sequence: Iterable,
    pattern_similarities: Iterable,
) -> pd.DataFrame:
    """Make dataframe with all recall metrics"""

    print("Calculating recall metrics...")
    metrics_df = pd.DataFrame()
    for n_trial in tqdm(range(num_trials - 1)):
        param = param_iterable[n_trial]
        items, times = np.unique(recall_sequence[n_trial], return_index=True)
        times = times[items != 0]
        times = np.sort(times)
        items = recall_sequence[n_trial][times]
        irts = np.insert(np.diff([times]), 0, 0)
        sizes = pattern_similarities[n_trial][items, items]
        ranks_all = np.array(
            [np.argsort(x).argsort() for x in pattern_similarities[n_trial]]
        )
        intersections = np.insert(
            pattern_similarities[n_trial][items[:-1], items[1:]], 0, 0
        )  #
        ranks = np.insert(ranks_all[items[:-1], items[1:]], 0, 0)
        dicttrial = {
            "items": items,
            "times": times,
            "trial": n_trial,
            "irts": irts,
            "sizes": sizes,
            "intersections": intersections,
            "ranks": ranks,
            "param": param,
        }


        # Avoid problematic files
        if (
            not dicttrial["items"].shape
            == dicttrial["times"].shape
            == dicttrial["irts"].shape
            == dicttrial["sizes"].shape
            == dicttrial["intersections"].shape
            == dicttrial["ranks"].shape
        ):
            continue

        trial_df = pd.DataFrame.from_dict(dicttrial)
        metrics_df = metrics_df.append(trial_df)

    metrics_df = metrics_df.reset_index().rename(columns={"index": "position"})
    num_memories_recalled = (
        metrics_df.groupby("trial")
        .agg({"position": np.max})
        .rename(columns={"position": "num_memories_recalled"})
    )
    metrics_df = metrics_df.merge(num_memories_recalled, on="trial")
    metrics_df.num_memories_recalled = (
        metrics_df.num_memories_recalled + 1
    )  # for 1-based indexing

    print("Done!")

    return metrics_df
