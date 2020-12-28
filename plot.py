"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Plotting functions.
"""

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import settings.paths as paths
import utils.file_loading as loading
from utils import recall_performance


def plot_parameter_sweep(
    parameter: str,
    sims_path: str,
    patterns_path: str,
    recalls_path: str,
    name_filter: str,
    title: str,
    xlabel: str,
    ylabel: str,
    file_name: str,
    panel_num: int,
) -> None:
    """Plots the parameter sweeps"""

    def plot(
        df: pd.DataFrame,
        title: str,
        xlabel: str,
        ylabel: str,
        panel_num: int,
        file_name: str,
    ) -> None:

        fig, axis = plt.subplots()
        axis.plot(df["param"], df["num_memories_recalled"], "-o", linewidth="0.5")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_yticks(range(16))

        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")

    print(f"Preparing {file_name} plot... ...")

    # File names will be parsed to get parameters
    file_names = sorted(
        loading.get_file_names_checked(
            {patterns_path, recalls_path},
            name_filter,
        )
    )

    file_paths_patterns = sorted(loading.get_file_paths(patterns_path, name_filter))
    file_paths_recalls = sorted(loading.get_file_paths(recalls_path, name_filter))

    assert len(file_names) == len(file_paths_patterns) == len(file_paths_recalls)
    num_trials = len(file_paths_patterns)

    parameters_used = recall_performance.get_used_parameters(
        file_names, sims_path, num_trials
    )

    recalls_sequence = recall_performance.load_sequences(file_paths_recalls)

    similarities = recall_performance.make_similarity(file_paths_patterns)

    metrics_df = recall_performance.get_main_df(
        num_trials,
        parameters_used[parameter],
        recalls_sequence,
        similarities,
    )
    # Calculate for the varying parameter
    metrics_df = metrics_df.groupby("param").apply(np.mean)

    plot(metrics_df, title, xlabel, ylabel, panel_num, file_name)


def plot_analysis(name_filter: str) -> None:
    """
    Plot all figures but for the parameter sweep.
    Takes advantage of only computing the dataframe once
    therefore it is a longer method.
    """

    print("Preparing temporal properties plots... ...")

    # File names will be parsed to get parameters
    file_names = sorted(
        loading.get_file_names_checked(
            {paths.PATTERNS_SEEDS_DIR, paths.RECALLS_SEEDS_DIR},
            name_filter,
        )
    )

    file_paths_patterns = sorted(
        loading.get_file_paths(paths.PATTERNS_SEEDS_DIR, name_filter)
    )
    file_paths_recalls = sorted(
        loading.get_file_paths(paths.RECALLS_SEEDS_DIR, name_filter)
    )

    assert len(file_names) == len(file_paths_patterns) == len(file_paths_recalls)
    num_trials = len(file_paths_patterns)

    parameters_used = recall_performance.get_used_parameters(
        file_names, paths.SIMS_SEEDS_DIR, num_trials
    )

    recalls_sequence = recall_performance.load_sequences(file_paths_recalls)

    similarities = recall_performance.make_similarity(file_paths_patterns)

    metrics_df = recall_performance.get_main_df(
        num_trials,
        parameters_used["seed"],
        recalls_sequence,
        similarities,
    )

    def plot_time_words_recalled(
        recalls_df: pd.DataFrame,
        panel_num: int,
        file_name: str,
    ) -> None:

        print("Plotting Average Words Recalled...")
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
        recalls_mean = np.mean(recalls_bincum, axis=0) * mid_memory
        recalls_mean = recalls_mean * mid_memory / 50
        tick_labels = np.arange(len(recalls_mean), dtype=int) * 100

        fig, axis = plt.subplots()
        axis.plot(recalls_mean)
        axis.set_title("Average Words Recalled")
        axis.set_xlabel("Time steps")
        axis.set_ylabel("Average words recalled")
        axis.set_xticklabels(tick_labels)

        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")
        print("Done!")

    plot_time_words_recalled(metrics_df, 0, "time_words_recalled.pdf")

    def plot_time_distribution_irt(
        recalls_df: pd.DataFrame, panel_num: int, file_name: str
    ) -> None:

        print("Plotting IRTs Distribution...")
        fig, axis = plt.subplots()

        axis.hist(recalls_df.irts.values, np.linspace(0, num_trials, 450))
        axis.set_title("IRTs Distribution")
        axis.set_xlabel("IRT value (cycles)")
        axis.set_ylabel("Count of IRT values")
        axis.set_xlim(0, 500)
        axis.set_yscale("log")
        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")
        print("Done!")

    plot_time_distribution_irt(metrics_df, 1, "time_distribution_irt.pdf")

    def plot_time_average_irt(
        recalls_df: pd.DataFrame,
        panel_num: int,
        file_name: str,
    ) -> None:

        df_fanout = recalls_df.groupby(["num_memories_recalled", "position"]).agg(
            {"irts": np.mean}
        )
        df_fanout = pd.pivot_table(
            data=recalls_df,
            index=["num_memories_recalled", "position"],
            values="irts",
            aggfunc=np.nanmean,
        ).reset_index()
        max_unique_recalls = []

        fig, axis = plt.subplots()
        for n_memory in range(16):
            data = df_fanout[df_fanout["num_memories_recalled"] == n_memory]
            data = data["irts"].to_numpy()
            max_unique_recalls.append(len(data))
            axis.plot(data, alpha=0.7)

        memory_range = range(max(max_unique_recalls))

        axis.set_title("Memory IRTs by Transition")
        axis.set_xlabel("Unique memory recalls")
        axis.set_ylabel("IRT value (cycles)")
        axis.set_xticks(memory_range)

        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")

    plot_time_average_irt(
        metrics_df,
        2,
        "time_average_irt.pdf",
    )

    def plot_probability_recall(
        recalls_df: pd.DataFrame,
        file_name: str,
        similarities: np.ndarray,
    ) -> None:

        print("Plotting Probability of Recall...")
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

        fig, axis = plt.subplots()

        axis.scatter(bins_centers, values_prob)
        axis.set_title("Probability of Recall")
        axis.set_xlabel("Memory size (neurons)")
        axis.set_ylabel("Probability of recall")

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")
        print("Done!")

    plot_probability_recall(metrics_df, "probability_recall.pdf", similarities)

    def plot_transitions_probability(
        recalls_df: pd.DataFrame, panel_num: int, file_name: str
    ) -> None:

        print("Plotting Probability of Transition...")
        values = recalls_df["ranks"][recalls_df["ranks"] > 0]
        memory_range = range(1, 15)
        counts = [np.sum(values == x) for x in memory_range]
        fig, axis = plt.subplots()

        axis.plot(range(1, 15), counts / np.sum(counts), "o-")
        axis.set_title("Probability of Transition")
        axis.set_xlabel("Transition size rank")
        axis.set_ylabel("Probability of transition")
        axis.set_xticks(memory_range)

        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")
        print("Done!")

    plot_transitions_probability(metrics_df, 0, "transitions_probability.pdf")

    def plot_transitions_irt(
        recalls_df: pd.DataFrame,
        panel_num: int,
        file_name: str,
    ) -> None:

        print("Plotting IRTs by Memory Similarity..")
        num_bins = 50
        intersections = recalls_df["intersections"][recalls_df["intersections"] > 0]
        irts = recalls_df["irts"][recalls_df["intersections"] > 0]
        bins = np.percentile(intersections, np.linspace(0, 100, num_bins + 1))
        intersections_bin = np.digitize(intersections, bins)
        irts_mean = [np.mean(irts[intersections_bin == x]) for x in range(num_bins)]
        bins_centers = (bins[:-1] + bins[1:]) / 2

        fig, axis = plt.subplots()

        axis.scatter(bins_centers, irts_mean)
        axis.set_title("IRTs by Memory Similarity")
        axis.set_xlabel("Memory similarity (neurons)")
        axis.set_ylabel("IRT value (cycles)")

        axis.text(
            -0.2,
            1.2,
            string.ascii_uppercase[panel_num],
            transform=axis.transAxes,
            size=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(paths.FIGURES_DIR, file_name))
        plt.close("all")
        print("Done!")

    plot_transitions_irt(metrics_df, 1, "transitions_irt.pdf")


def plot_all() -> None:
    """Run all plots"""

    plot_analysis(name_filter=".npy")

    # 0
    plot_parameter_sweep(
        parameter="noise_var",
        sims_path=paths.SIMS_NOISE_STD_DIR,
        patterns_path=paths.PATTERNS_NOISE_STD_DIR,
        recalls_path=paths.RECALLS_NOISE_STD_DIR,
        name_filter=".npy",
        title="Recalls Noise Variance",
        xlabel="Noise standard deviation",
        ylabel="Average words recalled",
        file_name="noise_var.pdf",
        panel_num=0,
    )

    # 1
    plot_parameter_sweep(
        parameter="cont_forth",
        sims_path=paths.SIMS_CONT_FORTH_DIR,
        patterns_path=paths.PATTERNS_CONT_FORTH_DIR,
        recalls_path=paths.RECALLS_CONT_FORTH_DIR,
        name_filter=".npy",
        title="Recalls Forward Contiguity",
        xlabel="Forward contiguity",
        ylabel="Average words recalled",
        file_name="cont_forth.pdf",
        panel_num=1,
    )

    # 2
    plot_parameter_sweep(
        parameter="cont_forth",
        sims_path=paths.SIMS_CONT_FORTH_LOW_DIR,
        patterns_path=paths.PATTERNS_CONT_FORTH_LOW_DIR,
        recalls_path=paths.RECALLS_CONT_FORTH_LOW_DIR,
        name_filter=".npy",
        title="Recalls Low Forward Contiguity",
        xlabel="Forward contiguity",
        ylabel="Average words recalled",
        file_name="cont_forth_low.pdf",
        panel_num=2,
    )


if __name__ == "__main__":
    plot_all()
