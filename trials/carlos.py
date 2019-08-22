import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image, plot_inhibition, \
    plot_weights

np.seterr(all='raise')
np.random.seed(512364)

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

for v in tqdm(range(n_pop)):
    for w in range(n_pop):

        raw_connectivity[v, w] = np.sum(
            (unique_patterns[v, :] - f)
            * (unique_patterns[w, :] - f)
        )

raw_connectivity *= relative_excitation

weights_without_inhibition = \
    raw_connectivity


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

def calculation(noise):

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


plot_weights(raw_connectivity, name='raw_connectivity')

phi_values = np.arange(phi_min, phi_max, 0.1)
current = np.zeros_like(phi_values)
noise = np.zeros_like(phi_values)

for phi in range(phi_values.size):
    pass
