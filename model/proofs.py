#%%
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(123)


NUM_NEURONS = 1  # 0 ** 5
NUM_MEMORIES = 16
# Activation
T_DECAY = 0.01
# Time
T_STEP = 0.001
T_TOT = 5
T_SIMULATED = int(T_TOT / T_STEP)
# Hebbian rule
EXCITATION = 13_000
SPARSITY = 0.1
# Gain
GAIN_THRESHOLD = 0
GAIN_EXP = 2 / 5
# Inhibition
SIN_MIN = 0.7 * EXCITATION
SIN_MAX = 1.06 * EXCITATION
# Noise
NOISE_STD = 65
# Forward and backward contiguity
CONT_FORTH = 1500 / EXCITATION
CONT_BACK = 400 / EXCITATION
# R
NUM_POINTS = T_TOT * 100

ITERS = 3  # CHANGE ME

#%% a


def get_log_int_sizes():
    def build_logical(prev_logical):
        """
        Stacks the previous logical array to itself, then adds an extra row at the bottom,
        with half zeros and half ones
        R: chaging this line practically does meanfield 
        """

        composite = np.hstack((prev_logical, prev_logical))
        try:
            half_size = prev_logical.shape[1]
        except:
            half_size = 1
        bottom = np.hstack((np.zeros(half_size), np.ones(half_size)))
        composite = np.vstack((composite, bottom))

        return composite

    # (1,)
    population_sizes = np.array(NUM_NEURONS)
    # (1,)
    log_int = np.array(0)

    for i in range(NUM_MEMORIES):
        # print("pop sizes:", population_sizes)
        new_sizes = (
            population_sizes * SPARSITY
        )  # R: calculates the sizes of the new intersctions
        # print("new sizes:", new_sizes)
        population_sizes = np.hstack(
            ((population_sizes - new_sizes), new_sizes)
        )  # R: updates the sizes
        # print("pop sizes:", population_sizes)

        log_int = build_logical(log_int)

        # print("=" * 10)
        if i == ITERS:
            break

    # (65536,), (17, 65536)
    return population_sizes, log_int


# pop_siz = population_sizes.T  # R: final vector of sizes


#%%


def get_connectivities(log_int):
    # ]  # R: final logical identity matrix
    # (65536, 16)
    connectivity_reg = (log_int.T)[:, 1:]
    # (65536, 16)
    connectivity_back = np.hstack(
        (
            np.zeros(connectivity_reg.shape[0])[:, None],
            connectivity_reg[:, : NUM_MEMORIES - 1],
        )
    )
    # (65536, 16)
    connectivity_forth = np.hstack(
        (connectivity_reg[:, 1:], np.zeros(connectivity_reg.shape[0])[:, None])
    )
    # 3 x (65536, 16)
    return connectivity_reg, connectivity_back, connectivity_forth


def prepare_times():
    # (500,)
    it_rec = np.linspace(1, T_SIMULATED, NUM_POINTS).astype(int)
    # (5001,)
    time = np.arange(start=1, stop=T_TOT + 1 + T_STEP, step=T_STEP)
    # (500,)
    t_rates = time[it_rec - 1] - 1
    return it_rec, time, t_rates


#%%


def osc(t):
    # (1,)
    return (SIN_MIN + SIN_MAX) / 2 + (SIN_MIN - SIN_MAX) / 2 * np.sin(
        2 * np.pi * t + np.pi / 2
    )


def noise(noise_std, population_sizes):
    # (65536,)
    return (
        noise_std
        / np.sqrt(population_sizes)
        * np.random.randn(population_sizes.shape[0])
        / np.sqrt(T_STEP)
    )


#%%


def gain(currents_vector):
    # (65536,)
    return (np.heaviside(currents_vector, 0) * currents_vector) ** 0.4


#%%


def get_initial_currents(num_memories, connectivity_reg):

    rnd_memory = 1  # np.random.randint(1, num_memories)
    currents_init = connectivity_reg[:, rnd_memory]
    currents_init += connectivity_reg[:, 0]

    # (65536,)
    return currents_init


#%% SE VIENE


def mult_W(Sgs, Vs, connectivity_reg, connectivity_back, connectivity_forth):
    # (16,)
    sparsity_vect = np.ones(NUM_MEMORIES) * SPARSITY
    # print((np.dot(connectivity_reg, Vs) - np.dot(sparsity_vect, Vs)- np.matmul(connectivity_reg, sparsity_vect) * np.sum(Sgs)+ np.dot(sparsity_vect, sparsity_vect) * np.sum(Sgs)+  np.matmul(CONT_FORTH* connectivity_forth, Vs))[:10])

    result = (
        connectivity_reg @ Vs  # (65536,) @ (16,) = (65536,)
        - sparsity_vect @ Vs  # (16,) @ (16,) = (1,)
        - connectivity_reg
        @ sparsity_vect
        * np.sum(Sgs)  # (65536, 16) @ (16,) * (1,) = (65536,)
        + sparsity_vect @ sparsity_vect * np.sum(Sgs)  # (16,) @ (16,) * (1,) = (1,)
        + CONT_FORTH * connectivity_forth @ Vs  # (1,) * (65536, 16) @ (16,) = (65536,)
        + CONT_BACK * connectivity_back @ Vs  # (1,) * (65536, 16) @ (16,) = (65536,)
    )

    return result


def evolv(
    t, curr, population_sizes, connectivity_reg, connectivity_back, connectivity_forth,
):
    # (65536,)
    act = gain(curr)
    # (65536,) = (65536,) * (65536,)
    sized_activations = population_sizes * act
    # (16,) = (16, 65536) @ (65536,)
    Vs = connectivity_reg.T @ sized_activations
    # (65536,)
    mult_W_ = mult_W(
        sized_activations, Vs, connectivity_reg, connectivity_back, connectivity_forth,
    )
    # (65536,)
    noise_ = noise(NOISE_STD, population_sizes)
    # (1,)
    sine = osc(t)

    # (1,) * ((1,) * (65536,) + (1,) * ( (1,) * (65536,) - (1,) * sum((65536,) * (65536,)) + (65536,)         )                       )
    result = (
        1
        / T_DECAY
        * (
            -1 * curr
            + 1
            / NUM_NEURONS
            * (EXCITATION * mult_W_ - sine * np.sum(population_sizes * act))
            + noise_
        )
    )

    # (65536,)
    return result


# def main():


# (65536,), (17, 65536)
population_sizes_, log_int_ = get_log_int_sizes()
# (1,)
num_pops = population_sizes_.shape[0]  # R: size of the actual simulated network

# 3 x (65536, 16)
connectivity_reg_, connectivity_back_, connectivity_forth_ = get_connectivities(
    log_int_
)


#%% Recanatesi's fun park


# Log_int as named data frame
log_int_df = pd.DataFrame(log_int_).astype(int)
log_int_df.index.name = "memories v"
log_int_df.columns.name = "pops >"
print(log_int_df)

#%%
# How many ones and zeros per pop (column)
pop_ones = log_int_df.sum()
pop_zeros = log_int_df.shape[0] - 1 - pop_ones

# Binomial distribution of zeros adjusted by number of ones
probs = pd.Series(
    pop_zeros.map(lambda x: sum(np.random.binomial(x, 0.1, 50000) == 0) / 50000),
    name="proba_bino",
)
probs /= 10 ** pop_ones
pop_sizes = pd.Series(population_sizes_, name="pop_sizes")
compared = pd.DataFrame((probs, pop_sizes))
print(compared)
