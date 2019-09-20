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

        # Inhibition
        self.phi = np.zeros(self.num_iter)
        self.inhibition = None

        # Memory patterns
        self.memory_patterns = np.zeros((self.n, self.p))
        self.v_pop = None
        self.n_per_pop = None
        self.s = None
        self.n_pop = None
        self.encoding = None

        # Connectivity
        self.raw_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.forward_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.backward_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.mu_forward = np.arange(self.p - 1)
        self.mu_backward = np.arange(1, self.p)

    def compute_phi(self):
        """

        :return:
        """
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
        """

        :return:
        """
        print("Computing memory patterns...")

        self.memory_patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.n, self.p))

        self.v_pop, self.n_per_pop = \
            np.unique([tuple(i) for i in self.memory_patterns], axis=0,
                      return_counts=True)

        self.s = self.n_per_pop / self.n

        self.n_pop = len(self.v_pop)

        print("Computing who is encoding what...")

        self.encoding = [
            (self.v_pop[:, mu] == 1).nonzero()[0] for mu in range(p)]

    def compute_connectivity(self):
        """

        :return:
        """
        print("XXXX")

        for v in tqdm(range(self.n_pop)):
            for w in range(self.n_pop):
                self.raw_connectivity[v, w] = np.sum(
                    (self.v_pop[v, :] - self.f)
                    * (self.v_pop[w, :] - self.f)
                )

                self.forward_connectivity[v, w] = np.sum(
                    self.v_pop[v, self.mu_forward] *
                    self.v_pop[w, self.mu_forward + 1]
                )

                self.backward_connectivity[v, w] = np.sum(
                    self.v_pop[v, self.mu_backward] *
                    self.v_pop[w, self.mu_backward - 1]
                )

    def add_factors(self):
        """

        :return:
        """
        print("XXXXX")

        self.raw_connectivity *= self.kappa
        self.forward_connectivity *= self.j_forward
        self.backward_connectivity *= self.j_backward
        self.inhibition *= self.kappa

    def compute_noise(self):
        """

        :return:
        """
        print("Computing uncorrelated Gaussian noise...")

        noise = np.zeros((self.n_pop, self.n_iter))

        for i in range(self.n_pop):
            noise[i] = \
                np.random.normal(loc=0,
                                 scale=(self.xi_0 * self.n_per_pop[i]) ** 0.5,
                                 size=self.n_iter) \
                / self.n_per_pop[i] * self.param_noise

    def initialize(self):
        """

        :return:
        """
        print("XXXXXXX")

        self.compute_phi()
        self.compute_memory_patterns()
        self.compute_connectivity()
        self.add_factors()
        self.compute_noise()





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

    network = Network()

    # Plots
    plot.activity_image(network.average_firing_rates_per_memory, dt=network.dt)
    plot.activity_curve(network.average_firing_rates_per_memory, dt=network.dt)
    plot.inhibition(network.inhibition, dt=network.dt)
    plot.phi(network.phi, dt=network.dt)
    plot.noise(network.noise, dt=network.dt)

    plot.weights(network.weights_without_inhibition,
                 name="weights_without_inhibition")
    plot.weights(network.raw_connectivity,
                 name="raw_connectivity")
    plot.weights(network.forward_connectivity,
                 name="forward_connectivity")
    plot.weights(network.backward_connectivity,
                 name="backward_connectivity")
    plot.current_curve(network.currents, dt=network.dt,
                       name="currents_population")
    plot.current_curve(network.currents_memory, dt=network.dt,
                       name="currents_memory")


if __name__ == '__main__':
    main()
