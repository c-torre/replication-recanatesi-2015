import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image, plot_inhibition, \
    plot_weights

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
phi_min = 0.70
phi_max = 1.06
tau_0 = 1
phase_shift = 0   # 0.5
# Short term association #
j_forward = 1500
j_backward = 400
# Time ###################
t_tot = 0.5  # 450!!!
dt = 0.001
# Noise #####
xi_0 = 65
# Initialization #########
r_ini = 1
first_p = 0  # memory presented first

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

# memory_patterns = []
# pbar = tqdm(total=n)
# while True:
#     pattern = \
#         np.random.choice([0, 1], p=[1 - f, f], size=p)
#     if not np.sum(pattern):
#         continue
#     memory_patterns.append(pattern)
#     pbar.update(1)
#     if len(memory_patterns) == n:
#         break
# pbar.close()

memory_patterns = np.asarray(memory_patterns)

unique_patterns_t, n_per_pattern = \
    np.unique([tuple(i) for i in memory_patterns], axis=1,
              return_counts=True)

v_pop = unique_patterns_t.T

s = n_per_pattern / n

n_pop = len(v_pop)

print("Compute who is encoding what...")

encoding = [
    (v_pop[:, mu] == 1).nonzero()[0] for mu in range(p)]

# print([len(encoding[i]) for i in range(len(encoding))])

# print("A priori")
#
# v_cardinal = np.zeros(n_pop)
# for i in range(n_pop):
#     for mu in range(p):
#         v_cardinal[i] += v_pop[i, mu]
#
# s_hat = np.zeros(n_pop)
# for i in range(n_pop):
#     s_hat[i] = (1-f) ** (p-v_cardinal[i]) * f**(v_cardinal[i])
#
# print("s hat", s_hat)
# print("s", s)
#
# fig, ax = plt.subplots()
#
# ax.scatter(np.arange(n_pop), s, color="C0", alpha=0.4)
# ax.scatter(np.arange(n_pop), s_hat, color="C1", alpha=0.4)
# plt.show()
#
# print(v_pop[0])
#
# print("v cardinal", v_cardinal)

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

raw_connectivity *= relative_excitation
forward_connectivity *= j_forward
backward_connectivity *= j_backward

weights_without_inhibition = \
    raw_connectivity \
    # + forward_connectivity \
    # + backward_connectivity

print("Computing uncorrelated Gaussian noise...")

# noise = np.zeros((n_pop, n_iteration))
#
# for i in range(n_pop):
#
#     noise[i] = \
#         np.random.normal(loc=0,
#                          scale=(xi_0 * s[i] * n) ** 0.5,
#                          size=n_iteration)

# noise = np.zeros((n_pop, n_iteration))

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
print("Size unique patterns", v_pop.shape)

print("Present pattern...")

# Update firing rates
firing_rates = np.zeros(n_pop)
firing_rates[encoding[first_p]] = r_ini

c_ini = r_ini ** (1/gamma) - theta
c = np.zeros(n_pop)
c[encoding[first_p]] = c_ini


print("t -1", "fr", np.average(firing_rates[encoding[first_p]],
                               weights=n_per_pattern[encoding[first_p]]))

print("Compute activation for each time step")

# For plot
average_firing_rates_per_memory = np.zeros((p, n_iteration))

for t in range(n_iteration):

    weights = weights_without_inhibition + inhibition[t]

    # Update current
    for v in range(n_pop):

        # Compute input
        input_v = np.sum(weights[v, :] * s[:] * firing_rates[:])

        if v == 0:
            # print("weights", weights[v, :])
            print("input", input_v)

        c[v] = \
            c[v] * (1 - dt) + \
            input_v * dt
            # (input_v + noise[v, t]) * dt

    # Update firing rates
    firing_rates[:] = 0
    cond = (c + theta) > 0
    # print(cond)
    firing_rates[cond] = (c[cond] + theta) ** gamma

    # print(firing_rates)
    # print("\n")
    # Store firing rate per memory

    print("t", t, "fr", np.average(firing_rates[encoding[first_p]],
                     weights=n_per_pattern[encoding[first_p]]))

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
# plot_noise(noise, dt=dt)
plot_weights(weights_without_inhibition, name='weights_without_inhibition')
plot_weights(raw_connectivity, name='raw_connectivity')
plot_weights(forward_connectivity, name='forward_connectivity')
plot_weights(backward_connectivity, name='backward_connectivity')
# plt.imshow(weights_without_inhibition)
# plt.imshow(weights)
