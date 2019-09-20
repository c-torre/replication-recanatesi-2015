"""
<[Re] Recanatesi (2015). Neural Network Model of Memory Retrieval>
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
The docstring for a module should generally list the classes, exceptions
and functions (and any other objects) that are exported by the module,
with a one-line summary of each. (These summaries generally give less
detail than the summary line in the object's docstring.) The docstring
for a package (i.e., the docstring of the package's __init__.py module)
should also list the modules and subpackages exported by the package.

:class Network: artificial neural network model of memory retrieval

"""

import os

import numpy as np
from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm

import tools.plots as plot
from tools.sinusoid import sinusoid

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)

np.seterr(all="raise")


class Network:
    """
    Hopfield network for memory retrieval based on the one in Recanatesi (2015)
    The docstring for a class should summarize its behavior and list the public
    methods and instance variables. If the class is intended to be subclassed,
    and has an additional interface for subclasses, this interface should be
    listed separately (in the docstring). The class constructor should be
    documented in the docstring for its __init__ method. Individual methods
    should be documented by their own docstring.

    """

    def __init__(self,
                 # Architecture
                 num_neurons=100000,
                 num_memories=16,
                 # Activation
                 t_decay=0.01,
                 # Gain
                 threshold=0,
                 gain_exp=2/5,
                 # Hebbian rule
                 excitation=13000,
                 sparsity=0.01,
                 # Inhibition
                 sin_min=0.7,
                 sin_max=1.06,
                 t_oscillation=1,
                 phase_shift=0.75,
                 # Short term association
                 cont_forward=1500,
                 cont_backward=400,
                 # Time
                 t_tot=14,
                 t_step=0.001,
                 # Noise
                 noise_var=65,
                 # Initialization
                 init_rate=1,
                 first_memory=7,
                 # Replication parameters
                 param_noise=10,
                 param_current=4.75
                 ):

        # Model parameters
        self.num_neurons = num_neurons
        self.num_memories = num_memories

        self.t_decay = t_decay

        self.threshold = threshold
        self.gain_exp = gain_exp

        self.excitation = excitation
        self.sparsity = sparsity

        self.sin_min = sin_min
        self.sin_max = sin_max
        self.t_oscillation = t_oscillation
        self.phase_shift = phase_shift

        self.cont_forward = cont_forward
        self.cont_backward = cont_backward

        self.t_tot = t_tot
        self.t_step = t_step

        self.noise_var = noise_var

        self.init_rate = init_rate
        self.first_memory = first_memory

        self.param_noise = param_noise  # Extra replication parameter
        self.param_current = param_current  # Extra replication parameter

        # General pre-computations
        self.num_iter = int(t_tot / t_step)
        self.relative_excitation = excitation / num_neurons
        self.time_param = t_step / t_decay

        # Inhibition
        self.sin = np.zeros(self.num_iter)
        self.inhibition = np.zeros(self.num_iter)

        # Memory patterns
        self.memory_patterns = np.zeros((self.num_neurons, self.num_memories))
        self.v_pop = None
        self.n_per_pop = None
        self.s = None
        self.n_pop = None
        self.encoding = None
        self.compute_memory_patterns()

        # Connectivity
        self.regular_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.forward_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.backward_connectivity = np.zeros((self.n_pop, self.n_pop))
        self.weights_without_inhibition = np.zeros((self.n_pop, self.n_pop))
        self.mu_forward = np.arange(self.num_memories - 1)
        self.mu_backward = np.arange(1, self.num_memories)

        # Noise
        self.noise = np.zeros((self.n_pop, self.num_iter))

        # Neuron dynamics
        self.firing_rates = np.zeros(self.n_pop)
        self.c = np.zeros(self.n_pop)
        self.average_firing_rates_per_memory = np.zeros((self.num_memories,
                                                         self.num_iter))
        self.currents = np.zeros((self.n_pop, self.num_iter))
        self.currents_memory = np.zeros((self.num_memories, self.num_iter))

        #####################
        # # Start network # #
        #####################
        self.start_network()

    def compute_memory_patterns(self):
        """

        :return:
        """

        print("Computing memory patterns...")

        self.memory_patterns = \
            np.random.choice([0, 1], p=[1 - self.sparsity, self.sparsity],
                             size=(self.num_neurons, self.num_memories))

        self.v_pop, self.n_per_pop = \
            np.unique([tuple(i) for i in self.memory_patterns], axis=0,
                      return_counts=True)

        self.s = self.n_per_pop / self.num_neurons

        self.n_pop = len(self.v_pop)

        print("Finding who is encoding what...")

        self.encoding = [
            (self.v_pop[:, mu] == 1).nonzero()[0] for mu in range(
                self.num_memories)]

    def compute_phi(self):
        """

        :return:
        """

        print("Calculating inhibition values...")

        for t in range(self.num_iter):
            self.sin[t] = sinusoid(
                min_=self.sin_min,
                max_=self.sin_max,
                period=self.t_oscillation,
                t=t,
                phase_shift=self.phase_shift * self.t_oscillation,
                dt=self.t_step
            )

        self.inhibition = - self.sin

    def compute_connectivity(self):
        """

        :return:
        """

        print("Adjusting regular, forward and backward connectivity...")

        for v in tqdm(range(self.n_pop)):
            for w in range(self.n_pop):
                self.regular_connectivity[v, w] = np.sum(
                    (self.v_pop[v, :] - self.sparsity)
                    * (self.v_pop[w, :] - self.sparsity)
                )

                self.forward_connectivity[v, w] = np.sum(
                    self.v_pop[v, self.mu_forward] *
                    self.v_pop[w, self.mu_forward + 1]
                )

                self.backward_connectivity[v, w] = np.sum(
                    self.v_pop[v, self.mu_backward] *
                    self.v_pop[w, self.mu_backward - 1]
                )

        self.regular_connectivity *= self.excitation
        self.forward_connectivity *= self.cont_forward
        self.backward_connectivity *= self.cont_backward
        self.inhibition *= self.excitation

        self.weights_without_inhibition = \
            self.regular_connectivity \
            + self.forward_connectivity \
            + self.backward_connectivity

    def compute_noise(self):
        """

        :return:
        """

        print("Calculating uncorrelated Gaussian noise...")

        for i in range(self.n_pop):
            self.noise[i] = \
                np.random.normal(loc=0,
                                 scale=(self.noise_var * self.n_per_pop[i])
                                 ** 0.5,
                                 size=self.num_iter) \
                / self.n_per_pop[i] * self.param_noise

    def initialize_neuron_dynamics(self):
        """

        :return:
        """

        print("Initializing neuron dynamics...")

        # Initialize firing rates
        self.firing_rates[self.encoding[self.first_memory]] = self.init_rate

        # initialize current
        c_ini = self.init_rate ** (1 / self.gain_exp) - self.threshold
        self.c[self.encoding[self.first_memory]] = c_ini

    def compute_neuron_dynamics(self):
        """

        :return:
        """

        print("Computing activation at each time step...")

        for t in tqdm(range(self.num_iter)):

            # Update current
            for v in range(self.n_pop):
                # Compute weights
                weights = (self.weights_without_inhibition[v, :]
                           + self.inhibition[t]) / self.num_neurons

                # Compute input
                input_v = np.sum(weights[:] * self.n_per_pop[:] *
                                 self.firing_rates[:])

                self.c[v] += self.time_param * (-self.c[v] + input_v
                                                + self.noise[v, t])

                self.currents[v, t] = self.c[v]

            # Update firing rates
            self.firing_rates[:] = 0
            cond = (self.c + self.threshold) > 0
            self.firing_rates[cond] = (self.c[cond] * self.param_current
                                       + self.threshold) ** self.gain_exp

            for mu in range(self.num_memories):
                fr = self.firing_rates[self.encoding[mu]]
                n_corresponding = self.n_per_pop[self.encoding[mu]]

                self.average_firing_rates_per_memory[mu, t] = \
                    np.average(fr, weights=n_corresponding)

                c_mu = self.c[self.encoding[mu]]

                self.currents_memory[mu, t] = \
                    np.average(c_mu, weights=n_corresponding)

    def start_network(self):
        """

        :return:
        """

        self.compute_phi()
        self.compute_connectivity()
        self.compute_noise()
        self.initialize_neuron_dynamics()
        self.compute_neuron_dynamics()


def main():
    """

    """

    print("<[Re] Recanatesi (2015)>  Copyright (C) <2019>\n"
          "<de la Torre-Ortiz C, Nioche A>\n"
          "This program comes with ABSOLUTELY NO WARRANTY.\n"
          "This is free software, and you are welcome to redistribute it\n"
          "under certain conditions; see the 'LICENSE' file for details.\n")
    np.random.seed(123)

    network = Network()

    # Plots
    print("Plotting...")
    plot.activity_image(network.average_firing_rates_per_memory,
                        dt=network.t_step)
    plot.activity_curve(network.average_firing_rates_per_memory,
                        dt=network.t_step)

    plot.inhibition(network.inhibition, dt=network.t_step)
    plot.phi(network.sin, dt=network.t_step)

    plot.noise(network.noise, dt=network.t_step)

    plot.weights(network.weights_without_inhibition,
                 name="weights_without_inhibition")
    plot.weights(network.regular_connectivity,
                 name="regular_connectivity")
    plot.weights(network.forward_connectivity,
                 name="forward_connectivity")
    plot.weights(network.backward_connectivity,
                 name="backward_connectivity")

    plot.current_curve(network.currents, dt=network.t_step,
                       name="currents_population")
    plot.current_curve(network.currents_memory, dt=network.t_step,
                       name="currents_memory")

    print("Done!")


if __name__ == '__main__':
    main()
