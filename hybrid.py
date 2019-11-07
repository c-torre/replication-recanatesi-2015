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
from tqdm import tqdm

import tools.sinusoid

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)

np.seterr(all="raise")

# Architecture
num_neurons = 100000
num_memories = 16
# Activation
t_decay = 0.01
# Gain
threshold = 0
gain_exp = 2 / 5
# Hebbian rule
excitation = 13000
sparsity = 0.01
# Inhibition
sin_min = 0.7
sin_max = 1.06
t_oscillation = 1
phase_shift = 0.75
# Short term association
cont_forth = 1500
cont_back = 400
# Time
t_tot = 14
t_step = 0.001
# Noise
noise_var = 65
# Initialization
init_rate = 1
first_memory = 7
# Replication parameters
param_noise = 10
param_current = 4.75


def compute_memory_patterns():
    """

    :return:
    """

    print("Computing memory patterns...")

    memory_patterns = \
        np.random.choice([0, 1], p=[1 - sparsity, sparsity],
                         size=(num_neurons, num_memories))

    v_pop, neurons_per_pop = \
        np.unique([tuple(i) for i in memory_patterns], axis=0,
                  return_counts=True)

    num_pops = len(v_pop)

    print("Finding who is encoding what...")

    neurons_encoding = [
        (v_pop[:, mu] == 1).nonzero()[0] for mu in range(
            num_memories)]
    return memory_patterns, v_pop, neurons_per_pop, num_pops, neurons_encoding


memory_patterns, v_pop, neurons_per_pop, num_pops, neurons_encoding = \
    compute_memory_patterns()

num_iter = int(t_tot / t_step)
relative_excitation = excitation / num_neurons
time_param = t_step / t_decay

# Inhibition
sine_wave = np.zeros(num_iter)

# Memory patterns
fraction_pop = None

# Connectivity
mu_forward = np.arange(num_memories - 1)
mu_backward = np.arange(1, num_memories)

# Noise
noise = np.zeros((num_pops, num_iter))

# Neuron dynamics
firing_rates = np.zeros([num_pops])
current = np.zeros([num_pops])
average_firing_rates_per_memory = np.zeros((num_memories, num_iter))
currents = np.zeros((num_pops, num_iter))
currents_memory = np.zeros((num_memories, num_iter))


def compute_phi():
    """

    :return:
    """

    print("Calculating inhibition values...")

    for t in range(num_iter):
        sine_wave[t] = tools.sinusoid.sinusoid(
            min_=sin_min,
            max_=sin_max,
            period=t_oscillation,
            t=t,
            phase_shift=phase_shift * t_oscillation,
            dt=t_step
        )

    return -sine_wave


inhibition = compute_phi()


def compute_connectivity(inh):
    """

    :return:
    """

    print("Computing regular, forward and backward connectivity...")

    reg_connectivity = np.zeros((num_pops, num_pops))
    forth_connectivity = np.zeros((num_pops, num_pops))
    back_connectivity = np.zeros((num_pops, num_pops))

    for v in tqdm(range(num_pops)):
        for w in range(num_pops):
            reg_connectivity[v, w] = np.sum(
                (v_pop[v, :] - sparsity)
                * (v_pop[w, :] - sparsity)
            )

            forth_connectivity[v, w] = np.sum(
                v_pop[v, mu_forward] *
                v_pop[w, mu_forward + 1]
            )

            back_connectivity[v, w] = np.sum(
                v_pop[v, mu_backward] *
                v_pop[w, mu_backward - 1]
            )

    reg_connectivity *= excitation
    forth_connectivity *= cont_forth
    back_connectivity *= cont_back
    inh *= excitation

    weights_without_inh = \
        reg_connectivity \
        + forth_connectivity \
        + back_connectivity

    return reg_connectivity, forth_connectivity, back_connectivity, \
        weights_without_inh


regular_connectivity, forward_connectivity, backward_connectivity, \
    weights_without_inhibition = compute_connectivity()


def compute_noise():
    """

    :return:
    """

    print("Calculating uncorrelated Gaussian noise...")

    for pop in range(num_pops):
        noise[pop] = \
            np.random.normal(loc=0,
                             scale=(noise_var * neurons_per_pop[pop]) ** 0.5,
                             size=num_iter) \
            / neurons_per_pop[pop] * param_noise

#####################
# # Start network # #
#####################
# start_network()

# class Network:
#     """
#     Hopfield network for memory retrieval based on the one in Recanatesi (2015)
#     The docstring for a class should summarize its behavior and list the public
#     methods and instance variables. If the class is intended to be subclassed,
#     and has an additional interface for subclasses, this interface should be
#     listed separately (in the docstring). The class constructor should be
#     documented in the docstring for its __init__ method. Individual methods
#     should be documented by their own docstring.
#
#     """
#
#
#
#
#
#     def initialize_neuron_dynamics():
#         """
#
#         :return:
#         """
#
#         print("Initializing neuron dynamics...")
#
#         # Initialize firing rates
#         firing_rates[
#             neurons_encoding[first_memory]] = init_rate
#
#         # initialize current
#         c_ini = init_rate ** (1 / gain_exp) - threshold
#         current[neurons_encoding[first_memory]] = c_ini
#
#     def compute_neuron_dynamics():
#         """
#
#         :return:
#         """
#
#         print("Computing activation at each time step...")
#
#         for t in tqdm(range(num_iter)):
#
#             # Update current
#             for pop in range(num_pops):
#                 # Compute weights
#                 weights = (weights_without_inhibition[pop, :]
#                            + inhibition[t]) / num_neurons
#
#                 # Compute input
#                 input_v = np.sum(weights[:] * neurons_per_pop[:] *
#                                  firing_rates[:])
#
#                 current[pop] += time_param * (
#                         -current[pop] + input_v
#                         + noise[pop, t])
#
#                 currents[pop, t] = current[pop]
#
#             # Update firing rates
#             firing_rates[:] = 0
#             cond = (current + threshold) > 0
#             firing_rates[cond] = (current[cond] * param_current
#                                        + threshold) ** gain_exp
#
#             for mu in range(num_memories):
#                 fr = firing_rates[neurons_encoding[mu]]
#                 n_corresponding = neurons_per_pop[
#                     neurons_encoding[mu]]
#
#                 average_firing_rates_per_memory[mu, t] = \
#                     np.average(fr, weights=n_corresponding)
#
#                 c_mu = current[neurons_encoding[mu]]
#
#                 currents_memory[mu, t] = \
#                     np.average(c_mu, weights=n_corresponding)
#
#     def start_network():
#         """
#
#         :return:
#         """
#
#         compute_phi()
#         compute_connectivity()
#         compute_noise()
#         initialize_neuron_dynamics()
#         compute_neuron_dynamics()
#
#
# def plot_all(network):
#     print("Plotting...")
#     plot.attractors(network.average_firing_rates_per_memory,
#                     dt=network.t_step)
#     plot.firing_rates(network.average_firing_rates_per_memory,
#                       dt=network.t_step)
#
#     plot.inhibition(network.inhibition, dt=network.t_step)
#     plot.sine_wave(network.sine_wave, dt=network.t_step)
#     plot.simplified_sine_wave(network.sine_wave, dt=network.t_step)
#
#     plot.noise(network.noise, dt=network.t_step)
#     print(network.noise.shape)
#
#     plot.weights(network.weights_without_inhibition,
#                  type_="no_inhibition")
#     plot.weights(network.regular_connectivity,
#                  type_="regular")
#     plot.weights(network.forward_connectivity,
#                  type_="forward")
#     plot.weights(network.backward_connectivity,
#                  type_="backward")
#
#     plot.currents(network.currents, dt=network.t_step,
#                   type_="population")
#     plot.currents(network.currents_memory, dt=network.t_step,
#                   type_="memory")
#
#     print("Done!")
#
#
# def main():
#     """
#
#     """
#
#     print("<[Re] Recanatesi (2015)>  Copyright (C) <2019>\n"
#           "<de la Torre-Ortiz C, Nioche A>\n"
#           "This program comes with ABSOLUTELY NO WARRANTY.\n"
#           "This is free software, and you are welcome to redistribute it\n"
#           "under certain conditions; see the 'LICENSE' file for details.\n")
#
#     np.random.seed(123)
#
#     network = Network()
#
#     plot_all(network)
#
#
# if __name__ == '__main__':
#     main()
