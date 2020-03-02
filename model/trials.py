#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(123)

tqdm.pandas()

NUM_NEURONS = 100000
NUM_MEMORIES = 16
T_STEP = 0.01
T_TOT = 10
# Hebbian rule
EXCITATION = 13_000
SPARSITY = 0.0
# Gain
GAIN_THRESHOLD = 0
GAIN_EXP = 2 / 5
# Inhibition
SIN_MIN = 0.7
SIN_MAX = 1.0

#%%

neurons_encoding_memories = np.random.choice(
    [0, 1], size=(NUM_NEURONS, NUM_MEMORIES), p=[1 - SPARSITY, SPARSITY]
)
#%%

populations_encoding_memories, population_sizes = np.unique(
    [tuple(i) for i in neurons_encoding_memories], axis=0, return_counts=True
)

#%%

population_sizes_encoding_memories = (
    populations_encoding_memories * population_sizes[:, None]
)

#%%

# Get the connectivity matrix
connectivity_regular = np.matmul(
    (populations_encoding_memories - SPARSITY),
    (populations_encoding_memories.T - SPARSITY),
)

#%%inhibition


def sine_wave(time):
    return np.sin(2 * np.pi * time + np.pi / 2)


#%%

times_vector = np.arange(start=0, stop=T_TOT, step=T_STEP)
sine_vector = np.vectorize(sine_wave)(times_vector)
sine_vector = ((SIN_MIN - SIN_MAX) / 2) + ((SIN_MIN - SIN_MAX) / (2 * sine_vector))
inhibition = sine_vector * EXCITATION / NUM_NEURONS  # is there a '-' missing?

#%%

# # %% Allocate weights tensor; this may be RAM-heavy hahah nope
# weights = np.multiply((connectivity_regular[None, :, :]).astype(np.float16),
# (inhibition[:, None, None]).astype(np.float16))


#%%

weights_forward = weights_backward = 0
weights_all = weights_regular + weights_forward + weights_backward


#%%

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#%%
pops = pd.DataFrame(
    array,
    index=[f"pop_{pop}" for pop in range(array.shape[0])],
    columns=[f"memory_{num_memory}" for num_memory in range(array.shape[1])],
)


def rer_roll(series):
    return pops.rolling(1).apply(lambda x: x.multiply(series))


# %%
