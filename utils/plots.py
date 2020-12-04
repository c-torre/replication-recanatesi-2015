"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Plotting functions.
"""

import os
import string

import matplotlib.pyplot as plt
import numpy as np
from settings import paths
from tqdm import tqdm


def plot_firing_rates_attractors(
    firing_rates: np.ndarray, t_step: float, num_oscillations: int, panel_num: int
) -> None:
    """Plot attractor (memory) number active per time cycle."""

    print("Plotting Active Attractor...")
    t_tot_plot = int(num_oscillations / t_step)
    firing_rates = firing_rates[:, :t_tot_plot]

    fig, axis = plt.subplots()

    num_memories = firing_rates.shape[0]

    heatmap = axis.imshow(
        firing_rates,
        interpolation="none",
        extent=[0, num_oscillations, num_memories - 0.5, -0.5],
    )

    axis.set_title("Attractors")
    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Attractor number")
    axis.set_xticks(range(num_oscillations))
    axis.set_yticks(range(num_memories))

    fig.colorbar(heatmap, ax=axis, label="Average firing rate")

    axis.set_aspect(aspect="auto")
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[panel_num],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(paths.FIGURES_DIR, "firing_rates_attractors.pdf"))
    plt.close("all")
    print("Done!")


def plot_lines(
    array: np.ndarray,
    t_step: float,
    num_oscillations: int,
    panel_num: int,
    title: str,
    y_label: str,
    file_name: str,
) -> None:
    """Plot firing rates of each memory."""

    print(f"Plotting {title}...")
    t_tot_plot = int(num_oscillations / t_step)
    array = array[:, :t_tot_plot]
    num_memories = array.shape[0]
    tick_labels = np.hstack(([0], np.arange(0, num_memories, 2)))

    fig, axis = plt.subplots()

    for array_row in tqdm(array):
        axis.plot(array_row, linewidth=0.5, alpha=0.8)

    axis.set_title(title)
    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel(y_label)
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

def plot_weights(connectivity: np.ndarray,  panel_num: int, title: str, file_name: str):

    print(f"Plotting {title}...")
    fig, axis = plt.subplots()
    heatmap = axis.imshow(connectivity)

    axis.set_title(title)
    axis.set_xlabel("Population $i$")
    axis.set_ylabel("Population $j$")

    fig.colorbar(heatmap, ax=axis)
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
