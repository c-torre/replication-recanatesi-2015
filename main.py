import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image, plot_inhibition, \
    plot_weights

np.seterr(all='raise')
np.random.seed(1234)

# Architecture ###########
n = 100000
p = 16
# Activation #############
tau = 0.01
# Gain ###################
theta = 0
gamma = 2/5
# Hebbian rule ###########
kappa = 13000
f = 0.01
# Inhibition #############
phi_min = 0.70
phi_max = 1.06
tau_0 = 1
phase_shift = 0  #  0.5
# Short term association #
j_forward = 1500
j_backward = 400
# Time ###################
t_tot = 2  # 450!!!
dt = 0.001
# Noise #####
xi_0 = 65
# Initialization #########
r_ini = 1
first_p = 1  # memory presented first

# General pre-computations
n_iteration = int(t_tot / dt)
relative_excitation = kappa / n

print("Computing oscillatory inhibition values...")

phi = np.zeros(n_iteration)
for t in range(n_iteration):
    phi[t] = sinusoid(
        min_=phi_min,
        max_=phi_max,
        period=tau_0,
        t=t,
        phase_shift=phase_shift*tau_0,
        dt=dt
    )

inhibition = - phi * relative_excitation * p  # * 3000
# CHANGED took out relative excitation
# reasoning --> weights

print("Compute memory patterns...")

# Unique simplified network attributes
memory_patterns = \
    np.random.choice([0, 1], p=[1 - f, f],
                     size=(p, n))
unique_patterns_t, n_per_pattern = \
    np.unique([tuple(i) for i in memory_patterns], axis=1,
              return_counts=True)

unique_patterns = unique_patterns_t.T

s = n_per_pattern / n

n_pop = len(unique_patterns)

print("Compute who is encoding what...")

encoding = [
    (unique_patterns[:, mu] == 1).nonzero()[0] for mu in range(p)]

# print([len(encoding[i]) for i in range(len(encoding))])

print("Computing weights without inhibition...")

raw_connectivity = np.zeros((n_pop, n_pop))
forward_connectivity = np.zeros((n_pop, n_pop))
backward_connectivity = np.zeros((n_pop, n_pop))

mu_forward = np.arange(p-1)
mu_backward = np.arange(1, p)

for v in tqdm(range(n_pop)):
    for w in range(n_pop):

        raw_connectivity[v, w] = np.sum(
            (unique_patterns[v, :] - f)
            * (unique_patterns[w, :] - f)
        )

        forward_connectivity[v, w] = np.sum(
            unique_patterns[v, mu_forward] *
            unique_patterns[w, mu_forward+1]
        )

        backward_connectivity[v, w] = np.sum(
            unique_patterns[v, mu_backward] *
            unique_patterns[w, mu_backward - 1]
        )

raw_connectivity *= relative_excitation
forward_connectivity *= j_forward
backward_connectivity *= j_backward

weights_without_inhibition = \
    raw_connectivity \
    # + forward_connectivity \
    # + backward_connectivity

print("Computing uncorrelated Gaussian noise...")

# noise = np.zeros((n_pop, n_iteration))

# for i in range(n_pop):
#
#     noise[i] = \
#         np.random.normal(loc=0,
#                          scale=(xi_0 * n_per_pattern[i]) ** 0.5,
#                          size=n_iteration)

noise = np.zeros((n_pop, n_iteration))

for i in range(n_pop):

    noise[i] = \
        np.random.normal(loc=0,
                         scale=(xi_0 * s[i]) ** 0.5,
                         size=n_iteration)


#mean = 0.5
#span = 0.5
#noise = np.random.uniform(mean - span, mean + span, (n_pop, n_iteration))
# noise[:] *= n_per_pattern
# noise = np.random.normal(loc=0.0, scale=.003, size=(n_pop, n_iteration))

print("\n\nBasic info")
print("-" * 10)
print("P", p)
print("N pop", n_pop)
print("relative excitation", relative_excitation)
print("Size unique patterns", unique_patterns.shape)

print("Present pattern...")

# Update firing rates
firing_rates = np.zeros(n_pop)
firing_rates[encoding[first_p]] = r_ini

print(firing_rates)
print(np.average(firing_rates[encoding[first_p]]))

print("Compute activation for each time step")

# Initialize currents
c = np.zeros(n_pop)

# For plot
average_firing_rates_per_memory = np.zeros((p, n_iteration))

for t in tqdm(range(n_iteration)):

    weights = weights_without_inhibition + inhibition[t]

    # Update current
    for v in range(n_pop):

        # Compute input
        input_v = np.sum(weights[v, :] * s[:] * firing_rates[:])

        c[v] = \
            c[v] * (1 - dt) + \
            (input_v + noise[v, t]) * dt

    # Update firing rates
    firing_rates[:] = 0
    cond = (c + theta) > 0
    # print(cond)
    firing_rates[cond] = (c[cond] + theta) ** gamma

    # print(firing_rates)
    # print("\n")
    # Store firing rate per memory
    for mu in range(p):

        fr = firing_rates[encoding[mu]]
        n_corresponding = n_per_pattern[encoding[mu]]

        # print("fr", fr)
        # print("n corresponding", n_corresponding)
        #
        average_firing_rates_per_memory[mu, t] = \
            np.average(fr, weights=n_corresponding)


# Make plots
plot_activity_image(average_firing_rates_per_memory, dt=dt)
plot_activity_curve(average_firing_rates_per_memory, dt=dt)
plot_inhibition(inhibition, dt=dt)
plot_phi(phi, dt=dt)
plot_noise(noise, dt=dt)
plot_weights(weights_without_inhibition, name='weights_without_inhibition')
plot_weights(raw_connectivity, name='raw_connectivity')
plot_weights(forward_connectivity, name='forward_connectivity')
plot_weights(backward_connectivity, name='backward_connectivity')
# plt.imshow(weights_without_inhibition)
# plt.imshow(weights)
