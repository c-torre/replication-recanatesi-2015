"""
Orchestrate pipeline
"""

#%%

import os
import pickle

import paths
from recall_analysis.data_processing import combined
from recall_analysis.data_processing.pre_processing import intersections, recalls

BKP_DIR = paths.BKP_DIR

FILE_PATHS = {
    "seeds": paths.SEED_RESULTS_DIR,
    # "cont": paths.CONT_RESULTS_DIR,
    # "noise": paths.NOISE_RESULTS_DIR,
}
#%%


def make_pickles():
    # Intersections
    intersection_sizes = intersections.make_all()
    file_pkl = os.path.join(BKP_DIR, "intersection_sizes.p")
    pickle.dump(intersection_sizes, open(file_pkl, "wb"))
    # Recalls
    recalls_analysis_data_frames_results = {
        results_type: recalls.make_all(files_path)
        for results_type, files_path in FILE_PATHS.items()
    }

    for results_type, data_frames in recalls_analysis_data_frames_results.items():
        file_pkl = os.path.join(BKP_DIR, f"recalls_frames_{results_type}.p")
        pickle.dump(data_frames, open(file_pkl, "wb"))

    # Combined
    combined_recalls_intersections_all = combined.make_all(
        recalls_analysis_data_frames_results["seeds"], intersection_sizes
    )
    file_pkl = os.path.join(BKP_DIR, "combined_recalls_intersections.p")
    pickle.dump(combined_recalls_intersections_all, open(file_pkl, "wb"))
    # combined_recalls_intersections_all = {
    #     results_type: combined.make_all(recalls_analysis_data_frames, intersection_sizes)
    #     for results_type, recalls_analysis_data_frames in recalls_analysis_data_frames_results.items()
    # }

    # for results_type, data_frames in recalls_analysis_data_frames_results.items():
    #     file_pkl = os.path.join(BKP_DIR, f"combined_{results_type}.p")
    #     pickle.dump(data_frames, open(file_pkl, "wb"))


# NOT FOR J AND NOISE U BRAINLET
# make_pickles()


# combined_recalls_intersections_all = combined.make_all(recalls_analysis_data_frames)


# %%
