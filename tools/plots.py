"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Plotting functions.

Contains the functions to plot currents, firing rates, attractors, the sine
wave, inhibition, weights and noise.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import string

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)


def currents(current_values, type_, fig_num, dt=1.):
    """
    Plot current values.

    :param current_values: array-like
    :param type_: str
    :param fig_num: int
    :param dt: float
    """

    fig, ax = plt.subplots()

    titles = {
        "population": {"title": "Population Currents",
                       "filename": "currents_populations"},
        "memory": {"title": "Memory Currents",
                   "filename": "currents_memory"}}

    n_iteration = current_values.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt

    for i, y in enumerate(current_values):
        ax.plot(x, y, linewidth=0.5, alpha=1)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Average current")
    ax.set_title(titles[type_]["title"])
    ax.text(-0.2, 1.2, string.ascii_uppercase[fig_num],
            transform=ax.transAxes,
            size=20, weight='bold')

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, titles[type_]["filename"] + ".pdf"))


def firing_rates(f_rates, dt=1.):
    """
    Plot firing rates.

    :param f_rates: array-like
    :param dt: float
    """

    fig, ax = plt.subplots()

    n_iteration = f_rates.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt
    ys = f_rates

    for i, y in enumerate(ys):
        ax.plot(x, y, linewidth=0.5, alpha=1)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Average firing rate")
    ax.set_title("Firing Rates")
    ax.text(-0.2, 1.2, string.ascii_uppercase[0],
            transform=ax.transAxes,
            size=20, weight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "firing_rates.pdf"))


def attractors(f_rates, dt=1.):
    """
    Plot attractor number active per time cycle.

    :param f_rates: array_like
    :param dt: float
    """

    fig, ax = plt.subplots()

    n_memory, n_iteration = f_rates.shape

    im = ax.imshow(f_rates, cmap="jet",
                   extent=[0, n_iteration * dt, n_memory - 0.5, -0.5])

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Attractor number")

    fig.colorbar(im, ax=ax)

    ax.set_aspect(aspect='auto')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Attractors")
    ax.text(-0.2, 1.2, string.ascii_uppercase[1],
            transform=ax.transAxes,
            size=20, weight='bold')

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "attractors.pdf"))


def sine_wave(sine_wave_values, dt=1.):
    """
    Plot the sine wave that drives inhibition.

    :param sine_wave_values: array-like
    :param dt: float
    """

    fig, ax = plt.subplots()

    n_iteration = len(sine_wave_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = sine_wave_values

    ax.plot(x, y)
    ax.set_title("Sine wave")
    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel(r"$\phi$")
    ax.text(-0.2, 1.2, string.ascii_uppercase[0],
            transform=ax.transAxes,
            size=20, weight='bold')

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "sine_wave.pdf"))


def inhibition(inhibition_values, dt=1.):
    """
    Plot the inhibition wave.

    :param inhibition_values: array-like
    :param dt: float
    """
    fig, ax = plt.subplots()

    n_iteration = len(inhibition_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = inhibition_values

    ax.plot(x, y)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Inhibition")
    ax.set_title("Inhibition")
    ax.text(-0.2, 1.2, string.ascii_uppercase[1],
            transform=ax.transAxes,
            size=20, weight='bold')

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "inhibition.pdf"))


def weights(weights_array, type_, fig_num):
    """
    Plot weight matrix.

    :param weights_array: array-like
    :param type_: str
    :param fig_num: int
    """

    titles = {
        "no_inhibition": {"title": "Weights Without Inhibition",
                          "filename": "weights_without_inhibition"},
        "regular": {"title": "Regular Weights",
                    "filename": "weights_regular"},
        "forward": {"title": "Forward Weights",
                    "filename": "weights_forward"},
        "backward": {"title": "Backward Weights",
                     "filename": "weights_backward"}}

    fig, ax = plt.subplots()

    ax.set_xlabel(r"Population $i$")
    ax.set_ylabel(r"Population $j$")
    ax.set_title(titles[type_]["title"])
    ax.text(-0.2, 1.2, string.ascii_uppercase[fig_num],
            transform=ax.transAxes,
            size=20, weight='bold')

    im = ax.imshow(weights_array)

    plt.tight_layout()

    fig.colorbar(im, ax=ax)

    plt.savefig(os.path.join(FIG_DIR, titles[type_]["filename"] + ".pdf"))


def noise(noise_values, t_tot, dt):
    """
    Plot noise values.

    :param noise_values: array-like
    :param t_tot: int
    :param dt: float
    """

    fig, ax = plt.subplots()

    n_time_steps = int(t_tot / dt)

    x = np.arange(n_time_steps, dtype=float) * dt

    for y in noise_values:
        ax.plot(x, y[:n_time_steps], linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Noise")
    ax.set_title("Noise")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "noise.pdf"))


def probability_recall_given_size(neurons_encoding, probability_recall):
    x = sorted([np.amax(memory) for memory in neurons_encoding])

    fig, ax = plt.subplots()

    plt.tight_layout()
    ax.scatter(x, probability_recall)

    plt.savefig(os.path.join(FIG_DIR, "probability_recall_memory_size.pdf"))
