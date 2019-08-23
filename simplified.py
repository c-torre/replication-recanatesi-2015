import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image, plot_inhibition, \
    plot_weights, plot_current_curve

FIG_FOLDER = 'fig'
os.makedirs(FIG_FOLDER, exist_ok=True)

np.seterr(all='raise')
np.random.seed(123)

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
phi_min = 0.7   # 0.7  # 0.2  # 0.70
phi_max = 1.06   # 1.06
tau_0 = 1
phase_shift = 0.75   # 0.5
# Short term association #
j_forward = 0  # 1500!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
j_backward = 0  # 400!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Time ###################
t_tot = 4  # 450!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dt = 0.001
# Noise #####
xi_0 = 65
# Initialization #########
r_ini = 1
first_p = 7  # memory presented first

no_noise = False
no_fancy_connection = False

# General pre-computations
n_iter = int(t_tot / dt)
relative_excitation = kappa / n
time_param = dt/tau

print("Computing oscillatory inhibition values...")

phi = np.zeros(n_iter)
for t in range(n_iter):
    phi[t] = sinusoid(
        min_=phi_min,
        max_=phi_max,
        period=tau_0,
        t=t,
        phase_shift=phase_shift*tau_0,
        dt=dt
    )

inhibition = - phi

print("Compute memory patterns...")

memory_patterns = \
    np.random.choice([0, 1], p=[1 - f, f],
                     size=(n, p))

v_pop, n_per_pop = \
    np.unique([tuple(i) for i in memory_patterns], axis=0,
              return_counts=True)

# assert np.sum(v_pop[0]) == 0
# v_pop = v_pop[1:]
# n_per_pop = n_per_pop[1:]

s = n_per_pop / n

n_pop = len(v_pop)

print("Compute who is encoding what...")

encoding = [
    (v_pop[:, mu] == 1).nonzero()[0] for mu in range(p)]

print("Computing weights without inhibition...")

raw_connectivity = np.zeros((n_pop, n_pop))
forward_connectivity = np.zeros((n_pop, n_pop))
backward_connectivity = np.zeros((n_pop, n_pop))

mu_forward = np.arange(p-1)
mu_backward = np.arange(1, p)

for v in tqdm(range(n_pop)):
    for w in range(n_pop):

        raw_connectivity[v, w] = np.sum(
            (v_pop[v, :] - f)
            * (v_pop[w, :] - f)
        )

        forward_connectivity[v, w] = np.sum(
            v_pop[v, mu_forward] *
            v_pop[w, mu_forward + 1]
        )

        backward_connectivity[v, w] = np.sum(
            v_pop[v, mu_backward] *
            v_pop[w, mu_backward - 1]
        )

# Put factors ============================

raw_connectivity *= kappa
forward_connectivity *= j_forward
backward_connectivity *= j_backward
inhibition *= kappa

# ========================================

if no_fancy_connection:
    print("NO FANCY CONNECTION")
    forward_connectivity[:] = 0
    backward_connectivity[:] = 0

weights_without_inhibition = \
    raw_connectivity \
    + forward_connectivity \
    + backward_connectivity

print("Computing uncorrelated Gaussian noise...")

noise = np.zeros((n_pop, n_iter))

for i in range(n_pop):
    noise[i] = \
        np.random.normal(loc=0,
                         scale=(xi_0 * n_per_pop[i]) ** 0.5,
                         size=n_iter) \
        / n_per_pop[i]

if no_noise:
    print("NO NOISE")
    noise[:] = 0

print("\n\nBasic info")
print("-" * 10)
print("N pop", n_pop)
print("-" * 10)

print("Present pattern...")

# Initialize firing rates
firing_rates = np.zeros(n_pop)
firing_rates[encoding[first_p]] = r_ini

# initialize current
c = np.zeros(n_pop)
c_ini = r_ini ** (1/gamma) - theta
c[encoding[first_p]] = c_ini

print("Compute activation for each time step")

# For plot
average_firing_rates_per_memory = np.zeros((p, n_iter))

currents = np.zeros((n_pop, n_iter))

currents_memory = np.zeros((p, n_iter))

for t in tqdm(range(n_iter)):

    # Update current
    for v in range(n_pop):

        # Compute weights
        weights = (weights_without_inhibition[v, :] + inhibition[t]) / n

        # Compute input
        input_v = np.sum(weights[:] * n_per_pop[:] * firing_rates[:])

        # c[v] = \
        #     c[v] * (1 - time_param) + \
        #     (input_v + noise[v, t]) * time_param
        #     # input_v * time_param
        # # if c[v] < 0:
        # #     c[v] = 0

        c[v] += time_param * (
            -c[v] + input_v + noise[v, t] * 10
        )

        currents[v, t] = c[v]

    # Update firing rates
    firing_rates[:] = 0
    cond = (c + theta) > 0
    firing_rates[cond] = (c[cond] + theta) ** gamma

    for mu in range(p):

        fr = firing_rates[encoding[mu]]
        n_corresponding = n_per_pop[encoding[mu]]

        average_firing_rates_per_memory[mu, t] = \
            np.average(fr, weights=n_corresponding)

        c_mu = c[encoding[mu]]

        currents_memory[mu, t] = \
            np.average(c_mu, weights=n_corresponding)

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
plot_current_curve(currents, dt=dt, name="currents_population")
plot_current_curve(currents_memory, dt=dt, name="currents_memory")
