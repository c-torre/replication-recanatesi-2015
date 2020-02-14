"""
Path information relative to project root
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_and_make_dir_path(path_name_prev, path_name_next):
    directory = os.path.join(path_name_prev, path_name_next)
    os.makedirs(directory, exist_ok=True)
    return directory


#
RECALL_DIR = os.path.join(ROOT_DIR, "recall_analysis")
##
PROCESSING_RECALL_DIR = os.path.join(RECALL_DIR, "data_processing")
##
PLOTS_RECALL_DIR = os.path.join(RECALL_DIR, "plots")

#
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
##
CONT_RESULTS_DIR = os.path.join(RESULTS_DIR, "forward_contiguity")
##
NOISE_RESULTS_DIR = os.path.join(RESULTS_DIR, "noise")
##
SEED_RESULTS_DIR = os.path.join(RESULTS_DIR, "seed")
###
POPULATION_SIZES_DIR = os.path.join(SEED_RESULTS_DIR, "population_sizes")
###
POPULATIONS_MEMORIES_DIR = os.path.join(SEED_RESULTS_DIR, "populations_memories")
###
RECALLS_SEED_DIR = os.path.join(SEED_RESULTS_DIR, "recalls")
