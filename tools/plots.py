"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

The docstring for a module should generally list the classes, exceptions
and functions (and any other objects) that are exported by the module,
with a one-line summary of each. (These summaries generally give less
detail than the summary line in the object's docstring.) The docstring
for a package (i.e., the docstring of the package's __init__.py module)
should also list the modules and subpackages exported by the package.

:class Network: artificial neural network model of memory retrieval
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)


def sine_wave(sine_wave_values, dt=1.):
    """
    The docstring for a module should generally list the classes, exceptions
    and functions (and any other objects) that are exported by the module,
    with a one-line summary of each. (These summaries generally give less
    detail than the summary line in the object's docstring.) The docstring
    for a package (i.e., the docstring of the package's __init__.py module)
    should also list the modules and subpackages exported by the package.

    :class Network: artificial neural network model of memory retrieval
    """

    fig, ax = plt.subplots()

    n_iteration = len(sine_wave_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = sine_wave_values

    ax.plot(x, y)
    ax.set_title("Sine wave")
    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel(r"$\phi$")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "sine_wave.pdf"))


def simplified_sine_wave(sine_wave_values, dt=1.):
    fig, ax = plt.subplots(figsize=(6.4, 1))

    n_iteration = len(sine_wave_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = sine_wave_values

    ax.plot(x, y)

    ax.set_yticks(())

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\phi$")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "sine_wave_simplified.pdf"))


def noise(noise_values, stop_time=2, dt=1.):
    fig, ax = plt.subplots()

    n_iteration = noise_values.shape[1]

    time_range = stop_time * 1000

    x = np.arange(time_range, dtype=float) * dt

    for y in noise_values:
        ax.plot(x, y[:time_range], linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Noise")
    ax.set_title("Noise")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "noise.pdf"))


def firing_rates(f_rates, dt=1.):
    fig, ax = plt.subplots()

    n_iteration = f_rates.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt
    ys = f_rates

    for i, y in enumerate(ys):
        ax.plot(x, y, linewidth=0.5, alpha=1)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Average firing rate")
    ax.set_title("Firing Rates")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "firing_rates.pdf"))


def attractors(f_rates, dt=1.):
    fig, ax = plt.subplots()

    n_memory, n_iteration = f_rates.shape

    im = ax.imshow(f_rates, cmap="jet",
                   extent=[
                       0, n_iteration * dt,
                          n_memory - 0.5, -0.5
                   ])

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Attractor number")

    fig.colorbar(im, ax=ax)

    ax.set_aspect(aspect='auto')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Attractors")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "attractors.pdf"))


def inhibition(inhibition_values, dt=1.):
    fig, ax = plt.subplots()

    n_iteration = len(inhibition_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = inhibition_values

    ax.plot(x, y)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Inhibition")
    ax.set_title("Inhibition")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "inhibition.pdf"))


def weights(weights_array, type_):
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

    im = ax.imshow(weights_array)

    plt.tight_layout()

    fig.colorbar(im, ax=ax)

    plt.savefig(os.path.join(FIG_FOLDER, titles[type_]["filename"] + ".pdf"))


def currents(current_values, type_, dt=1.):
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

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, titles[type_]["filename"] + ".pdf"))

# def romani(activity):
#     fig, ax = plt.subplots()
#     im = ax.imshow(activity, aspect="auto",
#                    cmap="jet")
#     fig.colorbar(im, ax=ax)
#
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Memories")
#
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#
#     fig.tight_layout()
#
#     plt.show()
