"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

sinusoid: Sine wave function.
"""

import numpy as np


def sinusoid(min_, max_, period, t, phase_shift, dt=1.):
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
    frequency = (1 / period) * dt
    shift = min_ + amplitude

    return \
        amplitude \
        * np.sin(2 * np.pi * (t + phase_shift / dt) * frequency) \
        + shift
