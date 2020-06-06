"""
Generate a reproducible set of numpy seeds
"""

import os

import numpy as np

np.random.seed(123)


def generate_seeds(force=False):
    """ Generate and save 10**4 seeds, each to simulate one net """

    if not os.path.exists("seeds.npy") or force:
        print("Saving an array of seeds... ")
        np.save(
            os.path.join(".", "seeds"),
            np.random.randint(low=0, high=2 ** 32 - 1, size=10 ** 4),
        )
        print("Done!")
