"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Auxiliary functions.
"""

import os

import numpy as np
import pandas as pd
from settings import paths


def get_simulation_range_param(sim_mode: str, job_id: int, num_points: int) -> float:
    """Give single simulation range parameter"""

    def check_simulation_mode(sim_mode: str) -> None:
        """Sanitize parameter input for simulation mode"""

        assert sim_mode in ["seed", "cont_forth", "cont_forth_low", "noise_var"]

    def prepare_parameter_ranges() -> pd.DataFrame:
        """Load parameters CSV and normalize contiguity"""

        # Load CSV
        ranges = pd.read_csv(
            os.path.join(paths.PARAMETERS_DIR, "ranges.csv"), index_col=0
        )

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
        according to job ID.

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


def get_connectivities(populations: np.ndarray, num_memories: int) -> np.ndarray:
    """Make regular, forward, and backward connectivities."""

    connectivity_reg = populations
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


def oscillation(time: float, sin_min: float, sin_max: float) -> float:
    """Sine oscillation which drives inhibition."""

    return (sin_min + sin_max) / 2 + (sin_min - sin_max) / 2 * np.sin(
        2 * np.pi * time + np.pi / 2
    )


def gain(currents_vector: np.ndarray, gain_exp: float) -> np.ndarray:
    """Gain or activation function"""

    adaption = np.heaviside(currents_vector, 0)
    currents_vector *= adaption
    return currents_vector ** gain_exp




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




def prepare_times(t_tot: int, t_step: float):
    """Initialize the time vectors."""

    return np.arange(start=0, stop=t_tot + t_step, step=t_step)


def get_recall_sequence(rates: np.ndarray, recall_threshold: float) -> np.ndarray:
    """Get sequence of recalled memories"""

    # Select where inhibition minima are expected (1-indexed)
    rates_max = rates[:, ::100]
    rates_max[:, 0] = rates[:, 1]

    # Select only one memory: highert firing rate above threshold
    seq = np.argmax(rates_max, axis=0)
    seq_max = np.max(rates_max, axis=0)
    return seq * (seq_max > recall_threshold)


def get_dynamics_memories(
    array: np.ndarray,
    population_sizes: np.ndarray,
    patterns: np.ndarray,
    memories_similarities: np.ndarray,
) -> np.ndarray:
    """Get a dynamics attribute for each memory."""

    return (
        population_sizes * array.T @ patterns / np.diagonal(memories_similarities)
    ).T


def get_connectivity_term(
    connectivity: np.ndarray, excitator: float, num_neurons: int, sparsity: float
) -> np.ndarray:
    return (
        excitator
        / num_neurons
        * (connectivity - sparsity)
        @ (connectivity.T - sparsity)
    )


def oscillation_closure(
    func: callable, sin_min: float, sin_max: float, num_neurons: int
) -> callable:
    """Encapsulate parameters to vectorize function"""

    return lambda t_cycle: func(t_cycle, sin_min, sin_max) / num_neurons
