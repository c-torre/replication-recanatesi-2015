"""
Copyright (C) <2019-2020>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

Directory management.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#
PARAMETERS_DIR = os.path.join(BASE_DIR, "parameters")

#
SIMULATIONS_DIR = os.path.join(BASE_DIR, "simulations")
os.makedirs(SIMULATIONS_DIR, exist_ok=True)

# #
SIMS_SEEDS_DIR = os.path.join(SIMULATIONS_DIR, "seeds")
os.makedirs(SIMS_SEEDS_DIR, exist_ok=True)
# # #
RECALLS_SEEDS_DIR = os.path.join(SIMS_SEEDS_DIR, "recalls")
os.makedirs(RECALLS_SEEDS_DIR, exist_ok=True)
# # #
PATTERNS_SEEDS_DIR = os.path.join(SIMS_SEEDS_DIR, "patterns")
os.makedirs(PATTERNS_SEEDS_DIR, exist_ok=True)

# #
SIMS_CONT_FORTH_DIR = os.path.join(SIMULATIONS_DIR, "cont-forth")
os.makedirs(SIMS_CONT_FORTH_DIR, exist_ok=True)
# # #
RECALLS_CONT_FORTH_DIR = os.path.join(SIMS_CONT_FORTH_DIR, "recalls")
os.makedirs(RECALLS_CONT_FORTH_DIR, exist_ok=True)
# # #
PATTERNS_CONT_FORTH_DIR = os.path.join(SIMS_CONT_FORTH_DIR, "patterns")
os.makedirs(PATTERNS_CONT_FORTH_DIR, exist_ok=True)

# #
SIMS_CONT_FORTH_LOW_DIR = os.path.join(SIMULATIONS_DIR, "cont-forth-low")
os.makedirs(SIMS_CONT_FORTH_LOW_DIR, exist_ok=True)
# # #
RECALLS_CONT_FORTH_LOW_DIR = os.path.join(SIMS_CONT_FORTH_LOW_DIR, "recalls")
os.makedirs(RECALLS_CONT_FORTH_LOW_DIR, exist_ok=True)
# # #
PATTERNS_CONT_FORTH_LOW_DIR = os.path.join(SIMS_CONT_FORTH_LOW_DIR, "patterns")
os.makedirs(PATTERNS_CONT_FORTH_LOW_DIR, exist_ok=True)

# #
SIMS_NOISE_STD_DIR = os.path.join(SIMULATIONS_DIR, "noise-std")
os.makedirs(SIMS_NOISE_STD_DIR, exist_ok=True)
# # #
RECALLS_NOISE_STD_DIR = os.path.join(SIMS_NOISE_STD_DIR, "recalls")
os.makedirs(RECALLS_NOISE_STD_DIR, exist_ok=True)
# # #
PATTERNS_NOISE_STD_DIR = os.path.join(SIMS_NOISE_STD_DIR, "patterns")
os.makedirs(PATTERNS_NOISE_STD_DIR, exist_ok=True)


#
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
