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
T_SIMULATED = int(T_TOT / T_STEP)
# Hebbian rule
EXCITATION = 13_000
SPARSITY = 0.1
# Gain
GAIN_THRESHOLD = 0
GAIN_EXP = 2 / 5
# Inhibition
SIN_MIN = 0.7
SIN_MAX = 1.0
# Noise
NOISE_STD = 65

#%%

neurons_encoding_memories = np.random.choice(
    [0, 1], size=(NUM_NEURONS, NUM_MEMORIES), p=[1 - SPARSITY, SPARSITY]
)

#%%

populations_encoding_memories, population_sizes = np.unique(
    [tuple(i) for i in neurons_encoding_memories], axis=0, return_counts=True
)

num_populations = populations_encoding_memories.shape[0]

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

#%% Noise

noise = np.random.randn(T_SIMULATED, num_populations)

#%%

noise *= NOISE_STD / (
    T_STEP * population_sizes[None, :]
)  # The standard deviation is applied through multiplication, mean is 0

#%%

# # %% Allocate weights tensor; this may be RAM-heavy hahah nope
# weights = np.multiply((connectivity_regular[None, :, :]).astype(np.float16),
# (inhibition[:, None, None]).astype(np.float16))

population_fractions = (
    population_sizes / NUM_NEURONS
)  # Sv seems not used in origina code

#%%

# memory_init = np.random.randint(NUM_MEMORIES)  USE THIS
memory_init = 7
firing_rates_init = populations_encoding_memories[:, memory_init]

#%% Computation for each time step

activations = firing_rates_init**0.4  # ACTIVATIONS NOT NEEDED FIRST TIME STEP for **0.4
sized_activations = firing_rates_init * population_fractions

#%%

def activation


#%% loop

weights_iter = connectivity_regular * inhibition[0]
connectivity_component = np.dot(sized_activations, weights_iter)
prev_currents = firing_rates_init
noise_iter = noise[0]
#%%
currents_this_step = prev_currents + connectivity_component + noise_iter
#%% ##
currents = np.zeros((T_SIMULATED, num_populations))
#%%
with np.nditer(curents, flags=["multi_index"], op_flags=["readwrite"]) as it:
    for elem in it:#tqdm(it):
        if it
        # if it == 0:
        #     continue
        # elem[...] += currents[it.index]
        print(it.multi_index[0])
        stored = it.multi_index[0]
#%%

for num_iter, currents_iter in enumerate(tqdm(currents)):
    if num_iter == 0:
        currents[num_iter, :] = firing_rates_init
    weights_iter = connectivity_regular * inhibition[num_iter]
    prev_currents = currents[num_iter-1,:]
    activations = (prev_currents > 0).astype(int)**0.4
    sized_activations = prev_currents * population_sizes
    connectivity_component = np.dot(sized_activations, weights_iter)
    currents[num_iter, :] = prev_currents
    if num_iter == 100:
        break

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
