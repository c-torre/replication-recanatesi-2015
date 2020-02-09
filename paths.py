"""
Path information relative to project root
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


PLOTS_DIR = os.path.join(ROOT_DIR, "plt")
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

