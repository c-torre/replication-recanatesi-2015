#%%
import matplotlib.pyplot as plt
import matplotlib.style as mpstyle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

np.random.seed(123)

tqdm.pandas()

mpstyle.use("fast")


NUM_NEURONS = 10 ** 5
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
CONT_FORTH = 400 / EXCITATION
CONT_BACK = 1500 / EXCITATION
# R
NUM_POINTS = T_TOT * 100

#%%
# >>>>>>>>>>>>>>>>>>>>>>> Reca

population_sizes = np.array(NUM_NEURONS)
log_int = np.array(0)
sparsity_vect = np.ones(NUM_MEMORIES) * SPARSITY

#%%


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
    # print(composite)
    return composite


for i in range(NUM_MEMORIES):
    new_sizes = (
        population_sizes * SPARSITY
    )  # R: calculates the sizes of the new intersctions
    population_sizes = np.hstack(
        ((population_sizes - new_sizes), new_sizes)
    )  # R: updates the sizes

    log_int = build_logical(log_int)

#%%

pop_siz = population_sizes.T  # R: final vector of sizes

num_pops = population_sizes.shape[0]  # R: size of the actual simulated network

#%%

connectivity_back = (log_int.T)[
    :, :16
]  # R: final logical identity matrix; this is correct, not reg
connectivity_reg = np.hstack(
    (connectivity_back[:, 1:], np.zeros(connectivity_reg.shape[0])[:, None])
)
connectivity_forth = np.hstack(
    (connectivity_reg[:, 1:], np.zeros(connectivity_reg.shape[0])[:, None])
)

#%%

it_rec = np.linspace(1, T_SIMULATED, NUM_POINTS).astype(int)
time = np.arange(start=1, stop=T_TOT + 1 + T_STEP, step=T_STEP)
#%%

rates = np.zeros((num_pops, len(it_rec)))


#%%

t_rates = (
    time[it_rec - 1] - 1
)  # Matsad allows index syntax that unlseashes hell in a real language

#%%


def osc(t):
    return (SIN_MIN + SIN_MAX) / 2 + (SIN_MIN - SIN_MAX) / 2 * np.sin(
        2 * np.pi * t + np.pi / 2
    )


def noise(noise_std):
    return (
        noise_std
        / np.sqrt(population_sizes)
        * np.random.randn(num_pops)
        / np.sqrt(T_STEP)
    )


#%%


def gain(currents_vector):
    return (np.heaviside(currents_vector, 0) * currents_vector) ** 0.4


#%%


def fun(t, curr):
    return evolv(t, curr, gain(curr))


#%%

rnd_memory = np.random.randint(1, NUM_MEMORIES)
#%%
currents_init = connectivity_reg[:, rnd_memory]
currents_init += connectivity_reg[:, 0]

#%% SE VIENE


def mult_W(Sgs, Vs):
    # print(Vs.shape)
    # print((np.dot(connectivity_reg, Vs)).shape)
    return (
        # np.matmul(connectivity_reg, Vs)
        -np.dot(sparsity_vect, Vs)
        - np.matmul(connectivity_reg, sparsity_vect) * np.sum(Sgs)
        + np.dot(sparsity_vect, sparsity_vect) * np.sum(Sgs)
        + CONT_FORTH * np.matmul(connectivity_forth, Vs)
        + CONT_BACK * np.matmul(connectivity_back, Vs)
    )


def evolv(t, curr, act):
    sized_activations = population_sizes * act
    Vs = np.matmul(connectivity_reg.T, sized_activations)
    noise_ = noise(NOISE_STD)
    sine = osc(t)

    print(mult_W(sized_activations, Vs,))
    return (
        1
        / T_DECAY
        * (
            -1 * curr
            + 1
            / NUM_NEURONS
            * (
                EXCITATION * mult_W(sized_activations, Vs,)
                - sine * np.sum(population_sizes * act)
            )
            + noise_
        )
    )


ind_t = 0
currents = currents_init


for it in range(len(time)):
    currents += fun(time[it], currents) * T_STEP
    # print(currents[:10])
    # print(np.matmul(connectivity_reg.T ,(population_sizes * gain(currents))))
    if it in it_rec:
        ind_t += 1
        rates[:, ind_t] = gain(currents)
    break


#%%


# <<<<<<<<<<<<<<<<<<<<<<< Reca

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


def compute_sine_wave(sin_min, sin_max, excitation, t_tot, t_step):
    times_vector = np.arange(start=0, stop=t_tot, step=t_step)
    sine_vector = np.vectorize(sine_wave)(times_vector)
    amplitude = (sin_min - sin_max) / 2
    sine_vector = amplitude + amplitude * sine_vector
    inhibition = (
        sine_vector * excitation / NUM_NEURONS
    )  # is there a '-' missing? -- No, should be added in W calculation

    # inhibition *= 10000
    return sine_vector, inhibition


sine_vector, inhibition = compute_sine_wave(
    SIN_MIN * EXCITATION, SIN_MAX * EXCITATION, EXCITATION, T_TOT, T_STEP
)

#%%
# plt.plot(sine_vector)
# print(inhibition)


#%% Noise


def compute_noise(t_simulated, num_populations):

    noise = np.random.randn(t_simulated, num_populations)

    noise *= NOISE_STD / (
        T_STEP * population_sizes[None, :] / NUM_NEURONS
    )  # The standard deviation is applied through multiplication, mean is 0
    return noise


noise = compute_noise(T_SIMULATED, num_populations)

# plt.imshow(noise)
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

# activations = firing_rates_init**0.4  # ACTIVATIONS NOT NEEDED FIRST TIME STEP for **0.4
# sized_activations = firing_rates_init * population_fractions

# # #%%


# 1/tau*(-1*s+1/N*(J1*mult_W((S.*gs),Iv*(S.*gs))-osc(t)*sum(S.*gs))+noise(noise_amp))
# #%% loop

# weights_iter = connectivity_regular * inhibition[0]
# connectivity_component = np.dot(sized_activations, weights_iter)
# prev_currents = firing_rates_init
# noise_iter = noise[0]
# #%%
# currents_this_step = prev_currents + connectivity_component + noise_iter
# #%% ##
# currents = np.zeros((T_SIMULATED, num_populations))


def gain(currents_vector):
    return (np.heaviside(currents_vector, 0) * currents_vector) ** 0.4


#%%

#%%%%%%%%

currents = np.zeros((T_SIMULATED, num_populations))
for num_iter, currents_iter in enumerate(tqdm(currents)):
    if num_iter == 0:
        currents[num_iter] = firing_rates_init
        continue
    # Previous currents used in current computation
    prev_currents = currents[num_iter - 1]
    # Compute connectivity component
    activations = gain(prev_currents)
    sized_activations = prev_currents * population_fractions
    weights_iter = (
        EXCITATION / NUM_NEURONS * (connectivity_regular - sine_vector[num_iter])
    )
    connectivity_component = np.dot(sized_activations, weights_iter)
    # Noise
    noise_iter = noise[num_iter]
    # Add all together
    currents[num_iter] = (
        prev_currents
        + T_STEP * (-prev_currents + connectivity_component + noise_iter) / T_DECAY
    )

#%%

firing_rates = gain(currents)
firing_rates_memories = np.matmul(firing_rates, populations_encoding_memories)
plt.imshow(np.rot90(firing_rates_memories))
# sns.heatmap(data=np.rot90(firing_rates_memories))
# weights_forward = weights_backward = 0
# weights_all = weights_regular + weights_forward + weights_backward


#%%

for current_line in np.rot90(currents):
    plt.plot(current_line)

    #%%

    #%%
    plot.currents(currents_memory, dt=t_step, type_="memory", fig_num=1)

    # plot.firing_rates(average_firing_rates_per_memory, t_step=T_STEP)
    # plot.attractors(average_firing_rates_per_memory, t_step=T_STEP)

    # plot.sine_wave(sine_wave, dt=t_step)
    # plot.inhibition(inhibition, dt=t_step)

    # plot.weights(weights_without_inhibition,
    #              type_="no_inhibition", fig_num=0)
    # plot.weights(regular_connectivity,
    #              type_="regular", fig_num=1)
    # plot.weights(forward_connectivity,
    #              type_="forward", fig_num=2)
    # plot.weights(backward_connectivity,
    #              type_="backward", fig_num=3)

    # plot.noise(noise, dt=t_step, t_tot=t_tot)

    # plot.probability_recall_given_size(
    #     pop_num_encoding_mem, probability_recall_memories
    # )
# array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# #%%
# pops = pd.DataFrame(
#     array,
#     index=[f"pop_{pop}" for pop in range(array.shape[0])],
#     columns=[f"memory_{num_memory}" for num_memory in range(array.shape[1])],
# )


# def rer_roll(series):
#     return pops.rolling(1).apply(lambda x: x.multiply(series))


# # %%
