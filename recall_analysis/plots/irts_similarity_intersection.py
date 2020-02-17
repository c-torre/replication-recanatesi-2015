#%%
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import recall_analysis.data_processing.main
import paths

file_pkl = os.path.join(paths.BKP_DIR, "combined_recalls_intersections.p")

if not os.path.exists(file_pkl):
    recall_analysis.data_processing.main.make_pickles()

combined_recalls_intersections_all = pickle.load(open(file_pkl, "rb"))

#%%
concatting = combined_recalls_intersections_all[0]
for idx in range(len(combined_recalls_intersections_all)):
    if idx == 0:
        continue
    concatting = concatting.append(combined_recalls_intersections_all[idx])


#%%
# mask = concatting["transition"]
a = concatting["average_irts"]
a *= concatting["transition"]
a= a.values
a = a[a.nonzero()[0]]
a = np.hstack(([0], a))
# a *= mask


# print(a)
# a = a.values
# a = a.nonzero()

#%%
b = concatting["transition_size"]
b *= concatting["transition"]
b = b.values
b = b[b.nonzero()[0]]

# %%
c = pd.DataFrame([a, b])

#%%

import matplotlib.pyplot as plt


fig, axis = plt.subplots()



axis.scatter(x=b, y=a)
axis.set_xlabel("Intersection size")
axis.set_ylabel("Average IRT")
axis.set_title("Average IRT with Intersection Size")
# ax = plt.scatter(x=b, y=a)
# ax = sns.scatterplot(data=c)
# ax.set(

#     xlabel="Inter-retrieval times (cycles)",
#     ylabel="Number of appereances",
#     title="IRTs Distribution",
# )

