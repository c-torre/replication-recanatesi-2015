"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Plotting functions.
"""

import os

from typing import Hashable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

import file_loader
import paths
import recall_performance


def plot_parameter_sweep(
    parameter: str,
    sims_path: str,
    patterns_path: str,
    recalls_path: str,
    name_filter: str,
    xlabel: str,
    ylabel: str,
    fig_file_path: str,
    fig_name: str,
) -> None:
    """Plots the parameter sweeps"""

    def plot(
        df: pd.DataFrame, xlabel: str, ylabel: str, fig_file_path: str, fig_name: str,
    ) -> None:

        plt.plot(dft["param"], dft["N_recalled"], "-o", linewidth="0.5")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks(range(16))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    print(f"Preparing {fig_name} plot... ...")

    # File names will be parsed to get parameters
    file_names = sorted(
        file_loader.get_file_names_checked({patterns_path, recalls_path}, name_filter,)
    )

    file_paths_patterns = sorted(file_loader.get_file_paths(patterns_path, name_filter))
    file_paths_recalls = sorted(file_loader.get_file_paths(recalls_path, name_filter))

    assert len(file_names) == len(file_paths_patterns) == len(file_paths_recalls)
    num_trials = len(file_paths_patterns)

    parameters_used = recall_performance.get_used_parameters(
        file_names, sims_path, num_trials
    )

    recalls_sequence = recall_performance.load_sequences(file_paths_recalls)

    similarities = recall_performance.make_similarity(file_paths_patterns, num_trials)

    dft = recall_performance.get_main_df(
        num_trials, parameters_used[parameter], recalls_sequence, similarities,
    )

    plot(dft, xlabel, ylabel, paths.FIGURES_DIR, fig_name)


plot_parameter_sweep(
    parameter="noise_var",
    sims_path=paths.SIMS_NOISE_STD_DIR,
    patterns_path=paths.PATTERNS_NOISE_STD_DIR,
    recalls_path=paths.RECALLS_NOISE_STD_DIR,
    name_filter=".npy",
    xlabel="Noise standard deviation",
    ylabel="Average words recalled",
    fig_file_path=paths.FIGURES_DIR,
    fig_name="noise_var.pdf",
)

plot_parameter_sweep(
    parameter="cont_forth",
    sims_path=paths.SIMS_CONT_FORTH_DIR,
    patterns_path=paths.PATTERNS_CONT_FORTH_DIR,
    recalls_path=paths.RECALLS_CONT_FORTH_DIR,
    name_filter=".npy",
    xlabel="Forward contiguity",
    ylabel="Average words recalled",
    fig_file_path=paths.FIGURES_DIR,
    fig_name="cont_forth.pdf",
)

plot_parameter_sweep(
    parameter="cont_forth",
    sims_path=paths.SIMS_CONT_FORTH_LOW_DIR,
    patterns_path=paths.PATTERNS_CONT_FORTH_LOW_DIR,
    recalls_path=paths.RECALLS_CONT_FORTH_LOW_DIR,
    name_filter=".npy",
    xlabel="Forward contiguity",
    ylabel="Average words recalled",
    fig_file_path=paths.FIGURES_DIR,
    fig_name="cont_forth_low.pdf",
)


def plot_analysis(
    sims_path: str,
    patterns_path: str,
    recalls_path: str,
    name_filter: str,
    fig_file_path: str,
) -> None:
    """
    Plot all figures but for the parameter sweep.
    Takes advantage of only computing the dataframe once
    therefore it is a longer method.
    """

    print("Preparing temporal properties plots... ...")

    # File names will be parsed to get parameters
    file_names = sorted(
        file_loader.get_file_names_checked({patterns_path, recalls_path}, name_filter,)
    )

    file_paths_patterns = sorted(file_loader.get_file_paths(patterns_path, name_filter))
    file_paths_recalls = sorted(file_loader.get_file_paths(recalls_path, name_filter))

    assert len(file_names) == len(file_paths_patterns) == len(file_paths_recalls)
    num_trials = len(file_paths_patterns)

    parameters_used = recall_performance.get_used_parameters(
        file_names, sims_path, num_trials
    )

    recalls_sequence = recall_performance.load_sequences(file_paths_recalls)

    similarities = recall_performance.make_similarity(file_paths_patterns, num_trials)

    dft = recall_performance.get_main_df(
        num_trials, parameters_used["seed"], recalls_sequence, similarities,
    )

    def plot_time_words_recalled(
        recalls_df: pd.DataFrame, fig_file_path: str, fig_name: str,
    ) -> None:

        mid_memory = 7
        data_sparse = np.ones_like(recalls_df.trial.values.astype(int))
        recalls_bin = csr_matrix(
            (
                data_sparse,
                (
                    recalls_df.trial.values.astype(int),
                    recalls_df.times.values.astype(int),
                ),
            )
        ).toarray()
        recalls_bincum = np.cumsum(recalls_bin, axis=1)
        recalls_mean = np.mean(recalls_bincum, axis=0)
        plt.plot(recalls_mean * mid_memory)
        plt.xlabel("Time steps")
        plt.ylabel("Average words recalled")
        plt.yticks(range(mid_memory))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_time_words_recalled(dft, fig_file_path, "time_words_recalled.pdf")

    def plot_time_distribution_irt(
        recalls_df: pd.DataFrame, fig_file_path: str, fig_name: str,
    ) -> None:

        plt.hist(recalls_df.irts.values, np.linspace(0, num_trials, 30))
        plt.yscale("log")
        plt.xlabel("IRT value")
        plt.ylabel("Count of IRT values")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_time_distribution_irt(dft, fig_file_path, "time_distribution_irt.pdf")

    def plot_time_average_irt(
        recalls_df: pd.DataFrame, fig_file_path: str, fig_name: str,
    ) -> None:

        df_fanout = recalls_df.groupby(["N_recalled", "position"]).agg(
            {"irts": np.mean}
        )
        df_fanout = pd.pivot_table(
            data=recalls_df,
            index=["N_recalled", "position"],
            values="irts",
            aggfunc=np.nanmean,
        ).reset_index()
        for i in range(16):
            data = df_fanout[df_fanout["N_recalled"] == i]
            plt.plot(data["irts"].values)

        plt.xlabel("Unique memories recalled")
        plt.ylabel("IRT value (cycles)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_time_average_irt(dft, fig_file_path, "time_average_irt.pdf")

    def plot_probability_recall(
        recalls_df: pd.DataFrame,
        fig_file_path: str,
        fig_name: str,
        similarities: np.ndarray,
    ) -> None:

        num_bins = 50
        sizes = similarities[:, range(16), range(16)]
        sizes_retrieved = recalls_df["sizes"][recalls_df["sizes"] > 0]
        bins = np.percentile(sizes_retrieved, np.linspace(0, 100, num_bins + 1))
        sizes_bin = np.digitize(sizes, bins)
        sizes_retrieved_bin = np.digitize(sizes_retrieved, bins)
        sizes_counts = [np.sum(sizes_bin == x) for x in range(num_bins)]
        sizesretrieved_counts = [
            np.sum(sizes_retrieved_bin == x) for x in range(num_bins)
        ]
        bins_centers = (bins[:-1] + bins[1:]) / 2
        values_prob = np.array(sizesretrieved_counts) / np.array(sizes_counts)
        plt.scatter(bins_centers, values_prob)
        plt.xlabel("Memory size (neurons)")
        plt.ylabel("Probability of recall")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_probability_recall(dft, fig_file_path, "probability_recall.pdf", similarities)

    def plot_transitions_probability(
        recalls_df: pd.DataFrame, fig_file_path: str, fig_name: str,
    ) -> None:

        values = recalls_df["ranks"][recalls_df["ranks"] > 0]
        counts = [np.sum(values == x) for x in range(1, 15)]
        plt.plot(range(1, 15), counts / np.sum(counts))
        plt.xlabel("Transition size rank")
        plt.ylabel("Probability of transition")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_transitions_probability(dft, fig_file_path, "transitions_probability.pdf")

    def plot_transitions_irt(
        recalls_df: pd.DataFrame, fig_file_path: str, fig_name: str,
    ) -> None:

        num_bins = 50
        intersections = recalls_df["intersections"][recalls_df["intersections"] > 0]
        irts = recalls_df["irts"][recalls_df["intersections"] > 0]
        bins = np.percentile(intersections, np.linspace(0, 100, num_bins + 1))
        intersections_bin = np.digitize(intersections, bins)
        irts_mean = [np.mean(irts[intersections_bin == x]) for x in range(num_bins)]
        bins_centers = (bins[:-1] + bins[1:]) / 2
        plt.scatter(bins_centers, irts_mean)
        plt.xlabel("Memories similarity")
        plt.ylabel("IRT value (cycles)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_file_path, fig_name))
        plt.close("all")

    plot_transitions_irt(dft, fig_file_path, "transitions_irt.pdf")


plot_analysis(
    sims_path=paths.SIMS_SEEDS_DIR,
    patterns_path=paths.PATTERNS_SEEDS_DIR,
    recalls_path=paths.RECALLS_SEEDS_DIR,
    name_filter=".npy",
    fig_file_path=paths.FIGURES_DIR,
)
