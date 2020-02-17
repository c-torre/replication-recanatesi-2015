import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import recall_analysis.data_processing.main
import paths

file_pkl = os.path.join(paths.BKP_DIR, f"recalls_frames_seeds.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

recalls_analysis_data_frames_all = pickle.load(open(file_pkl, "rb")