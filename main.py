import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image, plot_inhibition

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
phase_shift = 0.0
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

inhibition = - phi*relative_excitation*p * 3000
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

weights_without_inhibition = np.zeros((n_pop, n_pop))

for v in tqdm(range(n_pop)):
    for w in range(n_pop):

        sum_ = 0
        for mu in range(p):
            sum_ += \
                (unique_patterns[v, mu] - f) \
                * (unique_patterns[w, mu] - f)

        sum_forward = 0
        for mu in range(p - 1):
            sum_forward += \
                unique_patterns[v, mu] \
                * unique_patterns[w, mu + 1]

        sum_backward = 0
        for mu in range(1, p):
            sum_backward += \
                unique_patterns[v, mu] \
                * unique_patterns[w, mu - 1]

        weights_without_inhibition[v, w] = \
            relative_excitation * sum_ + sum_forward + sum_backward

print("Computing uncorrelated Gaussian noise...")

noise = np.zeros((n_pop, n_iteration))

for i in range(n_pop):

    noise[i] = \
        np.random.normal(loc=0,
                         scale=(xi_0 * n_per_pattern[i]) ** 0.5,
                         size=n_iteration)


print("\n\nBasic info")
print("-" * 10)
print("P", p)
print("N pop", n_pop)
print("relative excitation", relative_excitation)
print()

print("Present pattern...")

# Update firing rates
firing_rates = np.zeros(n_pop)
firing_rates[unique_patterns[:, first_p] == 1] = r_ini

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
    firing_rates[cond] = (c[cond] + theta) ** gamma

    # Store firing rate per memory
    for mu in range(p):
        encoding_mu = encoding[mu]

        average_firing_rates_per_memory[mu, t] = \
            np.sum(firing_rates[encoding_mu])


# Make plots
plot_activity_image(average_firing_rates_per_memory, dt=dt)
plot_activity_curve(average_firing_rates_per_memory, dt=dt)
plot_inhibition(inhibition, dt=dt)
# plot_phi(phi, dt=dt)
plot_noise(noise, dt=dt)
# plt.imshow(weights_without_inhibition)
# plt.imshow(weights)
