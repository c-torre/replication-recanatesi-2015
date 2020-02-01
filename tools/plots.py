"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Plotting functions.

Contains the functions to plot currents, firing rates, attractors, the sine
wave, inhibition, weights and noise.
"""

import os
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


def currents(current_values, type_, fig_num, t_step=1.0):
    """
    Plot current values.

    :param current_values: array-like
    :param type_: str
    :param fig_num: int
    :param dt: float
    """

    fig, axis = plt.subplots()

    titles = {
        "population": {
            "title": "Population Currents",
            "filename": "currents_populations",
        },
        "memory": {"title": "Memory Currents", "filename": "currents_memory"},
    }

    n_iteration = current_values.shape[1]

    times = np.arange(n_iteration, dtype=float) * t_step

    for _, data in enumerate(current_values):
        axis.plot(times, data, linewidth=0.5, alpha=1)

    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Average current")
    axis.set_title(titles[type_]["title"])
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[fig_num],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, titles[type_]["filename"] + ".pdf"))

    plt.close(fig)


def firing_rates(f_rates, t_step=1.0):
    """
    Plot firing rates.

    :param f_rates: array-like
    :param dt: float
    """

    fig, axis = plt.subplots()

    n_iteration = f_rates.shape[1]

    times = np.arange(n_iteration, dtype=float) * t_step
    n_data = f_rates

    for _, data in enumerate(n_data):
        axis.plot(times, data, linewidth=0.5, alpha=1)

    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Average firing rate")
    axis.set_title("Firing Rates")
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[0],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "firing_rates.pdf"))

    plt.close(fig)


def attractors(f_rates, t_step=1.0):
    """
    Plot attractor number active per time cycle.

    :param f_rates: array_like
    :param dt: float
    """

    fig, axis = plt.subplots()

    n_memory, n_iteration = f_rates.shape

    image = axis.imshow(
        f_rates, cmap="jet", extent=[0, n_iteration * t_step, n_memory - 0.5, -0.5]
    )

    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Attractor number")

    fig.colorbar(image, ax=axis)

    axis.set_aspect(aspect="auto")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title("Attractors")
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[1],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "attractors.pdf"))

    plt.close(fig)


def sine_wave(sine_wave_values, t_step=1.0):
    """
    Plot the sine wave that drives inhibition.

    :param sine_wave_values: array-like
    :param dt: float
    """

    fig, axis = plt.subplots()

    n_iteration = len(sine_wave_values)

    times = np.arange(n_iteration, dtype=float) * t_step
    data = sine_wave_values

    axis.plot(times, data)
    axis.set_title("Sine wave")
    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel(r"$\phi$")
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[0],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "sine_wave.pdf"))

    plt.close(fig)


def inhibition(inhibition_values, t_step=1.0):
    """
    Plot the inhibition wave.

    :param inhibition_values: array-like
    :param dt: float
    """
    fig, axis = plt.subplots()

    n_iteration = len(inhibition_values)

    times = np.arange(n_iteration, dtype=float) * t_step
    data = inhibition_values

    axis.plot(times, data)

    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Inhibition")
    axis.set_title("Inhibition")
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[1],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "inhibition.pdf"))

    plt.close(fig)


def weights(weights_array, type_, fig_num):
    """
    Plot weight matrix.

    :param weights_array: array-like
    :param type_: str
    :param fig_num: int
    """

    titles = {
        "no_inhibition": {
            "title": "Weights Without Inhibition",
            "filename": "weights_without_inhibition",
        },
        "regular": {"title": "Regular Weights", "filename": "weights_regular"},
        "forward": {"title": "Forward Weights", "filename": "weights_forward"},
        "backward": {"title": "Backward Weights", "filename": "weights_backward"},
    }

    fig, axis = plt.subplots()

    axis.set_xlabel(r"Population $i$")
    axis.set_ylabel(r"Population $j$")
    axis.set_title(titles[type_]["title"])
    axis.text(
        -0.2,
        1.2,
        string.ascii_uppercase[fig_num],
        transform=axis.transAxes,
        size=20,
        weight="bold",
    )

    image = axis.imshow(weights_array)

    plt.tight_layout()

    fig.colorbar(image, ax=axis)

    plt.savefig(os.path.join(FIG_DIR, titles[type_]["filename"] + ".pdf"))

    plt.close(fig)


def noise(noise_values, t_tot, t_step):
    """
    Plot noise values.

    :param noise_values: array-like
    :param t_tot: int
    :param dt: float
    """

    fig, axis = plt.subplots()

    n_time_steps = int(t_tot / t_step)

    times = np.arange(n_time_steps, dtype=float) * t_step

    for data in noise_values:
        axis.plot(times, data[:n_time_steps], linewidth=0.5, alpha=0.2)

    axis.set_xlabel("Time (cycles)")
    axis.set_ylabel("Noise")
    axis.set_title("Noise")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "noise.pdf"))

    plt.close(fig)


def probability_recall_given_size(neurons_encoding, probability_recall):
    """ Figure 4, but not correct """

    num_memories = sorted([np.amax(memory) for memory in neurons_encoding])

    fig, axis = plt.subplots()

    plt.tight_layout()
    axis.scatter(num_memories, probability_recall)

    plt.savefig(os.path.join(FIG_DIR, "probability_recall_memory_size.pdf"))

    plt.close(fig)
