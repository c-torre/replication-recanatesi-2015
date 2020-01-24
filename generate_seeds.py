"""
Generate a reproducible set of numpy seeds
"""

import os

import numpy as np

np.random.seed(123)

np.save(
    os.path.join(".", "seeds"), np.random.randint(low=0, high=2 ** 32 - 1, size=10 ** 4)
)
