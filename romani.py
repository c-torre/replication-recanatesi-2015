import numpy as np
from tqdm import tqdm

from tools.sinusoid import sinusoid
from tools.plot import plot_romani

np.seterr(all='raise')

# Parameters ===================================================

L = 16
N = 3000
T = 0.015
T_th = 45
T_j0 = 25
J_0_min = 0.7
J_0_max = 1.2
t_max = 1000
f = 0.1
phase_shift = 0.0
first_p = 0
D_th = 1.9 * T
seed = 123

# ==============================================================

phase_shift = phase_shift
first_p = first_p

J = np.zeros((N, N))
V = np.zeros(N)

xi = np.random.choice([0, 1], p=[1 - f, f], size=(N, L))

th = np.random.uniform(-T, T, size=N)

first_th = th.copy()

previous_V = np.zeros(N)

population_activity = np.zeros((L, t_max))

idx_neuron = np.arange(N)

n_factor = 1 / (N * f * (1-f))

print('Building the connections...')

for i in tqdm(range(N)):
    for j in range(N):
        sum_ = np.sum((xi[i, :] - f) * (xi[j, :] - f))
        J[i, j] = sum_ * n_factor

print('Present pattern...')

V[:] = xi[:, first_p]

print('Main loop...')
for t in tqdm(range(t_max)):

    # Update inhibition =======================================
    J_0 = sinusoid(
        min_=J_0_min,
        max_=J_0_max,
        period=T_j0,
        phase_shift=phase_shift * T_j0,
        t=t
    )

    # Update activity ==========================================

    previous_V = V.copy()

    second_sum_ = np.sum(previous_V)
    multiplier = J_0 / (N * f)

    for i in range(N):
        sum_ = np.sum(J[i, :] * previous_V[:])

        inside_parenthesis = \
            sum_ - multiplier * second_sum_ - th[i]

        V[i] = int(inside_parenthesis > 0)

    # Update threshold =========================================

    th[:] -= (th[:] - first_th[:] - D_th * previous_V[:]) / T_th

    # Save population activity ==================================

    for mu in range(L):
        sum_ = np.sum((xi[:, mu] - f) * V[:])
        population_activity[mu, t] = sum_ * n_factor

plot_romani(activity=population_activity)
