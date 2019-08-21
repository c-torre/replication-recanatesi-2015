import numpy as np
from tqdm import tqdm

from tools.sinusoid import sinusoid
from tools.plot import \
    plot_phi, plot_noise, \
    plot_activity_curve, plot_activity_image

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
t_tot_discrete = int(t_tot / dt)
relative_excitation = kappa / n

print("Computing oscillatory inhibition values...")

phi = np.zeros(t_tot_discrete)
for t in range(t_tot_discrete):
    phi[t] = sinusoid(
        min_=phi_min,
        max_=phi_max,
        period=tau_0,
        t=t,
        phase_shift=0,  # phase_shift * tau_0,
        dt=dt
    )

inhibition = - phi*relative_excitation*p

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

print("Computing connectivity matrix...")

connectivity_matrix = np.zeros((n_pop, n_pop))

for v in tqdm(range(n_pop)):
    for w in range(n_pop):

        sum_ = 0
        for mu in range(p):
            sum_ += \
                (unique_patterns[v, mu] - f) \
                * (unique_patterns[w, mu] - f)

        connectivity_matrix[v, w] = \
            relative_excitation * sum_

        if v == w:
            connectivity_matrix[v, w] = 0

print("Compute SAM connectivity...")

sam_connectivity = np.zeros((n_pop, n_pop))
for v in tqdm(range(n_pop)):

    for w in range(n_pop):

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

        sam_connectivity[v, w] = \
            j_forward * sum_forward \
            + j_backward * sum_backward

print("Compute weights")
weights = np.zeros((n_pop, n_pop, t_tot_discrete))
for v in tqdm(range(n_pop)):
    for w in range(n_pop):
        for t in range(t_tot_discrete):
            weights[v, w, t] = connectivity_matrix[v, w] \
                               + sam_connectivity[v, w] \
                               + inhibition[t]

print("Computing uncorrelated Gaussian noise...")
# amplitude = xi_0 * s * num_neurons

noise_values = np.zeros((n_pop, t_tot_discrete))

for i in range(n_pop):

    noise_values[i] = \
        np.random.normal(loc=0,
                         scale=(xi_0 * n_per_pattern[i]) ** 0.5,
                         size=t_tot_discrete)


print("Present pattern...")
c = np.zeros(n_pop)
c[unique_patterns[:, first_p] == 1] = r_ini

print("Compute activation for each time step")

firing_rates = np.zeros(n_pop)

average_firing_rates_per_memory = np.zeros((p, t_tot_discrete))

# Update firing rates
for v in range(n_pop):

    current_v = c[v]

    if current_v + theta > 0:
        fr_v = (current_v + theta) ** gamma
    else:
        fr_v = 0

    firing_rates[v] = fr_v


for t in tqdm(range(t_tot_discrete)):

    # Update current
    for v in range(n_pop):
        current_v = c[v]

        # First
        first_term = current_v * (1 - dt)

        # Second
        sum_ = 0
        for w in range(n_pop):

            current_w = c[w]

            sum_ += \
                weights[v, w, t] \
                * s[w] \
                * firing_rates[w]

        second_term = sum_ * dt

        # Third
        third_term = noise_values[v, t] * dt

        c[v] = \
            first_term + second_term + third_term

    # Update firing rates
    for v in range(n_pop):

        current_v = c[v]

        if current_v + theta > 0:
            fr_v = (current_v + theta) ** gamma
        else:
            fr_v = 0

        firing_rates[v] = fr_v

    for mu in range(p):
        encoding_mu = (unique_patterns[:, mu] == 1).nonzero()[0]
        if len(encoding_mu):
            fr_mu = firing_rates[encoding_mu]
            mean = np.average(
                fr_mu,
                weights=n_per_pattern[encoding_mu])
        else:
            mean = 0

        average_firing_rates_per_memory[mu, t] = mean


# Make plots
plot_activity_image(average_firing_rates_per_memory, dt=dt)
plot_activity_curve(average_firing_rates_per_memory, dt=dt)
plot_phi(phi, dt=dt)
plot_noise(noise_values, dt=dt)
