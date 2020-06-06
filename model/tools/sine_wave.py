"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

sinusoid: Sine wave function.
"""

import numpy as np


def sinusoid(min_, max_, period, time, phase_shift, t_step=1.0):
    """
    Sine wave function with extra parameters.

    :param min_: float
    :param max_: float
    :param period: float
    :param t: int
    :param phase_shift: float
    :param dt: float
    """

    amplitude = (max_ - min_) / 2
    frequency = (1 / period) * t_step
    shift = min_ + amplitude

    return (
        amplitude * np.sin(2 * np.pi * (time + phase_shift / t_step) * frequency)
        + shift
    )
