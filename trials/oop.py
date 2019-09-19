import os

import numpy as np
from tqdm import tqdm

import tools.plots as plot
from tools.sinusoid import sinusoid


class Network:
    def __init__(self,
                 # Architecture ###########
                 n=100000,
                 p=16,
                 # Activation #############
                 tau=0.01,
                 # Gain ###################
                 theta=0,
                 gamma=2/5,
                 # Hebbian rule ###########
                 kappa=13000,
                 f=0.01,
                 # Inhibition #############
                 phi_min=0.7,
                 phi_max=1.06,
                 tau_0=1,
                 phase_shift=0.75,
                 # Short term association #
                 j_forward=1500,
                 j_backward=400,
                 # Time ###################
                 t_tot=14,
                 dt=0.001,
                 # Noise ##################
                 xi_0=65,
                 # Initialization #########
                 r_ini=1,
                 first_p=7,
                 # Replication parameters #
                 param_noise=10,
                 param_current=4.75
                 ):

        # Model parameters
        self.n = n
        self.p = p

        self.tau = tau

        self.theta = theta
        self.gamma = gamma

        self.kappa = gamma
        self.f = f

        self.phi_min = phi_min
        self.phi_max = phi_max
        self.tau_0 = tau_0
        self.phase_shift = phase_shift

        self.j_forward = j_forward
        self.j_backward = j_backward

        self.t_tot = t_tot
        self.dt = dt

        self.xi_0 = xi_0

        self.r_ini = r_ini
        self.first_p = first_p

        self.param_noise = param_noise  # Allows change of limit cycle
        self.param_current = param_current  # Changes the height of the peaks

        # General pre-computations
        self.num_iter = int(t_tot / dt)
        self.relative_excitation = kappa / n
        self.time_param = dt / tau

        self.phi = np.zeros(self.num_iter)
        self.inhibition = None



    def compute_phi(self):
        print("Computing oscillatory inhibition values...")

        for t in range(self.num_iter):
            self.phi[t] = sinusoid(
                min_=self.phi_min,
                max_=self.phi_max,
                period=self.tau_0,
                t=t,
                phase_shift=self.phase_shift * self.tau_0,
                dt=self.dt
            )

        self.inhibition = - self.phi

    def compute_memory_patterns(self):
        print("Computing memory patterns...")

        memory_patterns = \
            np.random.choice([0, 1], p=[1 - f, f],
                             size=(n, p))

        v_pop, n_per_pop = \
            np.unique([tuple(i) for i in memory_patterns], axis=0,
                      return_counts=True)

        s = n_per_pop / n

        n_pop = len(v_pop)

        print("Computing who is encoding what...")

        encoding = [
            (v_pop[:, mu] == 1).nonzero()[0] for mu in range(p)]

    def initialize(self):
        """

        :return:
        """

        self.compute_phi()

# General pre-computations






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
        / n_per_pop[i] * param_noise

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

        c[v] += time_param * (-c[v] + input_v + noise[v, t])

        currents[v, t] = c[v]

    # Update firing rates
    firing_rates[:] = 0
    cond = (c + theta) > 0
    firing_rates[cond] = (c[cond] * param_current + theta) ** gamma

    for mu in range(p):

        fr = firing_rates[encoding[mu]]
        n_corresponding = n_per_pop[encoding[mu]]

        average_firing_rates_per_memory[mu, t] = \
            np.average(fr, weights=n_corresponding)

        c_mu = c[encoding[mu]]

        currents_memory[mu, t] = \
            np.average(c_mu, weights=n_corresponding)


def main():
    fig_folder = 'fig'
    os.makedirs(fig_folder, exist_ok=True)

    np.seterr(all='raise')
    np.random.seed(123)

    # Plots
    plot.activity_image(average_firing_rates_per_memory, dt=dt)
    plot.activity_curve(average_firing_rates_per_memory, dt=dt)
    plot.inhibition(inhibition, dt=dt)
    plot.phi(phi, dt=dt)
    plot.noise(noise, dt=dt)
    plot.weights(weights_without_inhibition, name='weights_without_inhibition')
    plot.weights(raw_connectivity, name='raw_connectivity')
    plot.weights(forward_connectivity, name='forward_connectivity')
    plot.weights(backward_connectivity, name='backward_connectivity')
    plot.current_curve(currents, dt=dt, name="currents_population")
    plot.current_curve(currents_memory, dt=dt, name="currents_memory")


if __name__ == '__main__':
    main()
