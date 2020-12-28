"""
<[Re] Recanatesi (2015). Neural Network Model of Memory Retrieval>
Copyright (C) <2020>  <de la Torre-Ortiz C, Nioche A>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.plots as plots
import utils.simulation as sim
from settings import paths

# -------------------------------------------------------------------------- #
# -------------------- Change parameters if needed here -------------------- #
# -------------------------------------------------------------------------- #

PATTERNS_DIR = paths.PATTERNS_CONT_FORTH_LOW_DIR  # paths. ...
RECALLS_DIR = paths.RECALLS_CONT_FORTH_LOW_DIR  # paths. ...

assert (
    PATTERNS_DIR is not None and RECALLS_DIR is not None
), "Please choose a saving directory"

try:
    JOB_ID = int(
        os.getenv("SLURM_ARRAY_TASK_ID")
    )  # Changes seed per cluster simulation
except:
    JOB_ID = 33  # Default seed for non-cluster use

np.random.seed(JOB_ID)

PARAMETERS_DF = pd.read_csv(
    os.path.join(paths.PARAMETERS_DIR, "simulation.csv"), index_col=0
)

NUM_NEURONS = int(PARAMETERS_DF.loc["num_neurons"].array[0])
NUM_MEMORIES = int(PARAMETERS_DF.loc["num_memories"].array[0])
# Activation
T_DECAY = PARAMETERS_DF.loc["t_decay"].array[0]
RECALL_THRESHOLD = PARAMETERS_DF.loc["recall_threshold"].array[0]
# Time
T_STEP = PARAMETERS_DF.loc["t_step"].array[0]
T_TOT = PARAMETERS_DF.loc["t_tot"].array[0]
T_SIMULATED = int(T_TOT // T_STEP)
# Hebbian rule
EXCITATION = PARAMETERS_DF.loc["excitation"].array[0]
SPARSITY = PARAMETERS_DF.loc["sparsity"].array[0]
# Gain
GAIN_THRESHOLD = PARAMETERS_DF.loc["gain_threshold"].array[0]
GAIN_EXP = PARAMETERS_DF.loc["gain_exp"].array[0]
# Inhibition
SIN_MIN = PARAMETERS_DF.loc["sin_min"].array[0] * EXCITATION
SIN_MAX = PARAMETERS_DF.loc["sin_max"].array[0] * EXCITATION
# Noise
NOISE_VAR = PARAMETERS_DF.loc["noise_var"].array[0]
# Forward and backward contiguity
CONT_FORTH = PARAMETERS_DF.loc["cont_forth"].array[0] / NUM_NEURONS
CONT_BACK = PARAMETERS_DF.loc["cont_back"].array[0] / NUM_NEURONS
# For parameter sweeps (uncomment to select)
# CONT_FORTH = sim.get_simulation_range_param("cont_forth", JOB_ID, 100) / NUM_NEURONS
CONT_FORTH = sim.get_simulation_range_param("cont_forth_low", JOB_ID, 100) / NUM_NEURONS
# NOISE_VAR = sim.get_simulation_range_param("noise_var", JOB_ID, 100)

# -------------------------------------------------------------------------- #
# ------------------------- END parameter changes -------------------------- #
# -------------------------------------------------------------------------- #

# Functions dependent on seed
def make_patterns(num_neurons: int, num_memories: int, sparsity: float) -> np.ndarray:
    """Build memory neural patterns according to sparsity."""

    return np.random.choice(
        (False, True), p=(1 - sparsity, sparsity), size=(num_neurons, num_memories)
    )


def get_noise(
    noise_var: int,
    population_sizes: np.ndarray,
    times: np.ndarray,
    num_populations: int,
) -> np.ndarray:
    """Computes noise for all time iterations."""

    std = noise_var / (population_sizes ** 0.5)
    return std * np.random.normal(size=(len(times), num_populations))


# Connectivity
patterns = make_patterns(NUM_NEURONS, NUM_MEMORIES, SPARSITY)
populations, population_sizes = sim.get_populations_and_sizes(patterns)
num_populations = population_sizes.shape[0]

connectivity_reg, connectivity_back, connectivity_forth = sim.get_connectivities(
    populations, NUM_MEMORIES
)

populations_sized = populations * population_sizes[:, None]
memories_similarities = populations_sized.T @ populations

# Dynamics
time = sim.prepare_times(T_TOT, T_STEP)
sparsity_vect = np.full(NUM_MEMORIES, SPARSITY)
initial_memory = np.random.choice(range(NUM_MEMORIES))
oscillation = np.vectorize(
    sim.oscillation_closure(sim.oscillation, SIN_MIN, SIN_MAX, NUM_NEURONS)
)(time)

currents = np.zeros((num_populations, len(time)))
firing_rates = np.zeros((num_populations, len(time)))
n_currents = connectivity_reg[:, initial_memory].astype(np.float)

noise = get_noise(NOISE_VAR, population_sizes, time, num_populations)  # long


# Simulation
for n_iter, t_cycle in tqdm(enumerate(time)):

    activations = sim.gain(n_currents.copy(), GAIN_EXP)
    sized_activations = population_sizes * activations
    total_activation = np.sum(sized_activations)

    # (NUM_MEMORIES,) = (num_populations,) @ (num_populations, NUM_MEMORIES)
    memory_activations = sized_activations @ connectivity_reg
    # (num_populations) = (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # - (NUM_MEMORIES,) @ (NUM_MEMORIES,) - (num_populations, NUM_MEMORIES)
    # @ (NUM_MEMORIES,) * () + (NUM_MEMORIES,) @ (NUM_MEMORIES,) * ()
    connectivity_term = (
        connectivity_reg @ memory_activations
        - sparsity_vect @ memory_activations
        - connectivity_reg @ sparsity_vect * total_activation
        + sparsity_vect @ sparsity_vect * total_activation
    )
    # (num_populations,) = () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # + () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    contiguity_term = (
        CONT_FORTH * connectivity_forth @ memory_activations
        + CONT_BACK * connectivity_back @ memory_activations
    )

    updated_currents = (
        T_STEP
        / T_DECAY
        * (
            -n_currents
            + EXCITATION / NUM_NEURONS * (connectivity_term + contiguity_term)
            - oscillation[n_iter] * total_activation
            + noise[n_iter] / np.sqrt(T_STEP)
        )
    )

    n_currents += updated_currents
    firing_rates[:, n_iter] = activations.copy()
    currents[:, n_iter] = n_currents.copy()


# Save simulation results
file_name = f"s{JOB_ID}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{int(NOISE_VAR)}"
populations_sized = (connectivity_reg * population_sizes[:, None]).T

memories_similarities = populations_sized @ connectivity_reg
firing_rates_memories = sim.get_dynamics_memories(
    firing_rates, population_sizes, connectivity_reg, memories_similarities
)

np.save(os.path.join(PATTERNS_DIR, file_name), memories_similarities)
np.save(
    os.path.join(RECALLS_DIR, file_name),
    sim.get_recall_sequence(firing_rates_memories, RECALL_THRESHOLD),
)


# Transform data and plot detailed dynamics only on selected seed
if JOB_ID == 33:

    # Dynamics
    print("Preparing to plot dynamics")
    currents_memories = sim.get_dynamics_memories(
        currents, population_sizes, connectivity_reg, memories_similarities
    )
    currents_populations = (population_sizes * currents.T).T
    print("Done!")

    plots.plot_firing_rates_attractors(firing_rates_memories, T_STEP, 15, 0)
    plots.plot_lines(
        firing_rates_memories,
        T_STEP,
        15,
        1,
        "Firing Rates",
        "Average firing rate",
        "firing_rates_lines.pdf",
    )

    plots.plot_lines(
        currents_populations,
        T_STEP,
        15,
        0,
        "Population Current",
        "Average current",
        "currents_populations.pdf",
    )
    plots.plot_lines(
        currents_memories,
        T_STEP,
        15,
        1,
        "Memory Current",
        "Average current",
        "currents_memories.pdf",
    )

    # Oscillation
    plots.plot_lines(
        oscillation[None, :], T_STEP, 15, 0, "Oscillation", "$\phi$", "oscillation.pdf"
    )
    plots.plot_lines(
        oscillation[None, :] * EXCITATION / NUM_NEURONS,
        T_STEP,
        15,
        1,
        "Inhibition",
        "Inhibition",
        "inhibition.pdf",
    )

    # Weights
    weights_reg = sim.get_connectivity_term(
        connectivity_reg, EXCITATION, NUM_NEURONS, SPARSITY
    )
    weights_back = sim.get_connectivity_term(
        connectivity_back, EXCITATION, NUM_NEURONS, SPARSITY
    )
    weights_forth = sim.get_connectivity_term(
        connectivity_forth, EXCITATION, NUM_NEURONS, SPARSITY
    )
    weigths_without_inhibition = weights_reg + weights_back + weights_forth

    plots.plot_weights(
        weigths_without_inhibition,
        0,
        "Weights Without Inhibition",
        "weights_without_inhibition.pdf",
    )
    plots.plot_weights(weights_reg, 1, "Regular Weights", "weights_reg.pdf")
    plots.plot_weights(weights_back, 2, "Backward Weights", "weights_back.pdf")
    plots.plot_weights(weights_forth, 3, "Forward Weights ", "weights_forth.pdf")

    # Noise
    plots.plot_lines(
        noise.T / T_STEP ** 0.5, T_STEP, 15, 0, "Noise", "Noise", "noise.png"
    )
