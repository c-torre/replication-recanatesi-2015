#%%
"""
<[Re] Recanatesi (2015). Neural Network Model of Memory Retrieval>
Copyright (C) <2020>  <de la Torre-Ortiz C, Nioche A>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import paths

PARAMETERS_DIR = paths.PARAMETERS_DIR
PATTERNS_DIR = paths.PATTERNS_SEEDS_DIR  # paths. ...
RECALLS_DIR = paths.RECALLS_SEEDS_DIR  # paths. ...

assert (
    PATTERNS_DIR is not None and RECALLS_DIR is not None
), "Please choose a saving directory"

try:
    job_id = int(
        os.getenv("SLURM_ARRAY_TASK_ID")
    )  # Changes seed per cluster simulation
except:
    job_id = 0  # Default seed for non-cluster use

np.random.seed(job_id)

PARAMETERS_DF = pd.read_csv(os.path.join(PARAMETERS_DIR, "simulation.csv"), index_col=0)

NUM_NEURONS = int(PARAMETERS_DF.loc["num_neurons"].array[0])
NUM_MEMORIES = int(PARAMETERS_DF.loc["num_memories"].array[0])
# Activation
T_DECAY = PARAMETERS_DF.loc["t_decay"].array[0]
# Time
T_STEP = PARAMETERS_DF.loc["t_step"].array[0]
T_TOT = 10  # PARAMETERS_DF.loc["t_tot"].array[0] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
# For parameter sweeps
# CONT_FORTH = get_simulation_range_param("cont_forth", job_id, 100)
# CONT_FORTH = get_simulation_range_param("cont_forth_low", job_id, 100)
# NOISE_VAR = get_simulation_range_param("noise_var", job_id, 100)


def get_simulation_range_param(sim_mode: str, job_id: int, num_points: int) -> float:
    """Give single simulation range parameter"""

    def check_simulation_mode(sim_mode: str) -> None:
        """Sanitize parameter input for simulation mode"""

        assert sim_mode in ["seed", "cont_forth", "cont_forth_low", "noise_var"]

    def prepare_parameter_ranges() -> pd.DataFrame:
        """Load parameters CSV and normalize contiguity"""

        # Load CSV
        ranges = pd.read_csv(os.path.join(PARAMETERS_DIR, "ranges.csv"), index_col=0)
        # Normalize contiguity
        ranges.loc["cont_forth"] /= NUM_NEURONS
        ranges.loc["cont_forth_low"] /= NUM_NEURONS

        return ranges

    def get_linespace_parameters(
        sim_mode: str,
        param_ranges_df: pd.DataFrame,
        num_points: int,
    ) -> np.array:
        """Return simulation parameters"""

        # Prepare variables
        min_ = param_ranges_df.loc[sim_mode, "min_"]
        max_ = param_ranges_df.loc[sim_mode, "max_"]

        # Check user input
        assert "min_" in param_ranges_df.columns and "max_" in param_ranges_df
        assert max_ > min_
        if sim_mode == "seed":
            assert num_points == max_

        return np.linspace(min_, max_, num_points)

    def get_param_from_range(param_linespace: np.array, job_id: int) -> int:
        """
        Get one parameter value within range
        according to job_id.

        Index resets when job_id == num_points
        num_points = 3: 0 1 2 0 1 2 0 1 2 ...
        """

        return param_linespace[job_id % len(param_linespace)]

    check_simulation_mode(sim_mode)
    ranges_df = prepare_parameter_ranges()
    range_param = get_linespace_parameters(
        sim_mode,
        ranges_df,
        num_points,
    )

    return get_param_from_range(range_param, job_id)


def get_connectivities(log_int: np.ndarray, num_memories: int) -> np.ndarray:
    """Make regular, forward, and backward connectivities"""

    connectivity_reg = log_int.T
    connectivity_back = np.hstack(
        (
            np.zeros(connectivity_reg.shape[0])[:, None],
            connectivity_reg[:, : num_memories - 1],
        )
    )
    connectivity_forth = np.hstack(
        (connectivity_reg[:, 1:], np.zeros(connectivity_reg.shape[0])[:, None])
    )
    return connectivity_reg, connectivity_back, connectivity_forth


def osc(t: float, sin_min, sin_max) -> float:
    """Sine oscillation which drives inhibition"""

    return (sin_min + sin_max) / 2 + (sin_min - sin_max) / 2 * np.sin(
        2 * np.pi * t + np.pi / 2
    )


def gain(currents_vector: np.ndarray, gain_exp: float) -> np.ndarray:
    """Gain or activation function"""

    adaption = np.heaviside(currents_vector, 0)
    currents_vector *= adaption
    return currents_vector ** gain_exp


def make_patterns(num_neurons: int, num_memories: int, sparsity: float) -> np.ndarray:
    """Build memory neural patterns according to sparsity"""

    return np.random.choice(
        (False, True), p=[1 - sparsity, sparsity], size=(num_neurons, num_memories)
    )


def get_populations_and_sizes(patterns: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Check which neurons encode for the same memories and
    group them into populations, saving how many neurons
    belong to each population (population sizes).
    """

    populations, population_sizes = np.unique(
        [tuple(i) for i in patterns], axis=0, return_counts=True
    )
    return populations, population_sizes


patterns = make_patterns(NUM_NEURONS, NUM_MEMORIES, SPARSITY)
populations, population_sizes = get_populations_and_sizes(patterns)
#%%
num_populations = population_sizes.shape[0]

connectivity_reg_, connectivity_back_, connectivity_forth_ = get_connectivities(
    populations.T, NUM_MEMORIES
)


def prepare_times(t_tot: int, t_step: float):
    """Initialize the time vectors"""

    return np.arange(start=0, stop=t_tot + t_step, step=t_step)


time = prepare_times(T_TOT, T_STEP)
#%%

initial_memory = np.random.choice(range(NUM_MEMORIES))
currents = connectivity_reg_[:, initial_memory].astype(np.float)

sparsity_vect = np.ones(NUM_MEMORIES) * SPARSITY
rates = np.zeros((num_populations, len(time)))


def get_noise(
    noise_var: int,
    population_sizes: np.ndarray,
    times: np.ndarray,
    num_populations: int,
) -> np.ndarray:
    """Computes noise for all time iterations"""

    std = noise_var / (population_sizes ** 0.5)
    return std * np.random.normal(size=(len(times), num_populations))


noise_ = get_noise(NOISE_VAR, population_sizes, time, num_populations)

for it, t in tqdm(enumerate(time)):

    activations = gain(currents.copy(), GAIN_EXP)
    sized_activations = population_sizes * activations
    total_activation = np.sum(sized_activations)

    # (NUM_MEMORIES,) = (num_populations,) @ (num_populations, NUM_MEMORIES)
    Vs = sized_activations @ connectivity_reg_
    # (num_populations) = (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # - (NUM_MEMORIES,) @ (NUM_MEMORIES,) - (num_populations, NUM_MEMORIES)
    # @ (NUM_MEMORIES,) * () + (NUM_MEMORIES,) @ (NUM_MEMORIES,) * ()
    connectivity_term = (
        connectivity_reg_ @ Vs
        - sparsity_vect @ Vs
        - connectivity_reg_ @ sparsity_vect * total_activation
        + sparsity_vect @ sparsity_vect * total_activation
    )
    # (num_populations,) = () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # + () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    contiguity_term = (
        CONT_FORTH * connectivity_forth_ @ Vs + CONT_BACK * connectivity_back_ @ Vs
    )

    sine = osc(t, SIN_MIN, SIN_MAX) / NUM_NEURONS

    evolv = (
        T_STEP
        / T_DECAY
        * (
            -currents
            + EXCITATION / NUM_NEURONS * (connectivity_term + contiguity_term)
            - sine * total_activation
            + noise_[it] / np.sqrt(T_STEP)
        )
    )

    currents += evolv
    rates[:, it] = activations

proj_attr = (connectivity_reg_ * population_sizes[:, None]).T
similarity = proj_attr @ connectivity_reg_
rate_avg = (population_sizes * rates.T @ connectivity_reg_ / np.diagonal(similarity)).T
file_name = f"s{job_id}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{int(NOISE_VAR)}"

#%%

# save recall patterns
np.save(os.path.join(PATTERNS_DIR, file_name), similarity)

#%%


def transform_file(rates_) -> np.ndarray:
    rates_max = rates_[:, ::100]
    rates_max[:, 0] = rates_[:, 1]
    seq = np.argmax(rates_max, axis=0)
    seq_max = np.max(rates_max, axis=0)
    seq2 = seq * (seq_max > 15)
    return seq2


# save recall sequence
np.save(os.path.join(RECALLS_DIR, file_name), transform_file(rate_avg))
