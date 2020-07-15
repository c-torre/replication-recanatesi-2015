# works
#%%
import glob
import os
import pickle

#%% Figure 3
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy.io import loadmat
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

# folder = "you_local_folder_of_choice"
filename = "sim_"

# files = glob.glob(folder + filename + "_*")
results_dir = "./results/"
files_names = os.listdir(results_dir)
file_paths = sorted([results_dir + file_name for file_name in files_names])
file_paths = file_paths
files = file_paths
print(files)

#%%
rates_list = []
similarity_list = []
for i_file, file_ in tqdm(enumerate(files)):
    print(i_file)
    print(file_)
    target = file_
    if os.path.getsize(target) > 0:
        with open(target, "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            rates = unpickler.load()
    #rates = pickle.load(open(file_, "rb"))
    rates_list.append(rates["rate_avg"])
    similarity_list.append(rates["similarity"])

#%%
# rates_np = np.concatenate(rates_list)
rates_np = np.stack(rates_list)
N_trials = rates_np.shape[0]
rates_npmax = rates_np[
    :, :, ::100
]  # This selects out the maximum values of the rates oscillation which occur every 100 steps
# rates_npmax = rates_np[:, ::100]
#%%
rates_npmax[:, :, 0] = rates_np[
    :, :, 1
]  # This is because the max of the first item is not at position 0 but at position 1

# rates_npmax[:, 0] = rates_np[:, 1]

sequence = np.argmax(rates_npmax, axis=1)
T_max = sequence.shape[1]
# T_max = sequence.shape[0]
sequence_max = np.max(rates_npmax, axis=1)
sequence = sequence * (sequence_max > 15)

similarities = np.stack(similarity_list)
# similarities = np.concatenate(similarity_list)
#%%

df = pd.DataFrame()
for i_trial in tqdm(range(N_trials)):
    items, times = np.unique(sequence[i_trial], return_index=True)
    times = times[items != 0]
    times = np.sort(times)
    items = sequence[i_trial][times]
    irts = np.insert(np.diff([times]), 0, 0)
    sizes = similarities[i_trial][items, items]
    ranksall = np.array([np.argsort(x).argsort() for x in similarities[i_trial]])
    intersections = np.insert(similarities[i_trial][items[:-1], items[1:]], 0, 0)
    ranks = np.insert(ranksall[items[:-1], items[1:]], 0, 0)
    dicttrial = {
        "items": items,
        "times": times,
        "trial": i_trial,
        "irts": irts,
        "sizes": sizes,
        "intersections": intersections,
        "ranks": ranks,
    }
    dftrial = pd.DataFrame.from_dict(dicttrial)
    df = df.append(dftrial)

#%%
df = df.reset_index().rename(columns={"index": "position"})
N_recalled = (
    df.groupby("trial")
    .agg({"position": np.max})
    .rename(columns={"position": "N_recalled"})
)
df = df.merge(N_recalled, on="trial")
df.N_recalled = df.N_recalled + 1  # This is to fix the fact that python is 0-based

#%%

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (10, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

plt.subplot(131)
data_sparse = np.ones_like(df.trial.values.astype(int))
recalls_bin = csr_matrix(
    (data_sparse, (df.trial.values.astype(int), df.times.values.astype(int)))
).toarray()
recalls_bincum = np.cumsum(recalls_bin, axis=1)
recalls_meant = np.mean(recalls_bincum, axis=0)
plt.plot(recalls_meant)
plt.xlabel("time in cycles")
plt.ylabel("average # words recalled")

plt.subplot(132)
plt.hist(df.irts.values, np.linspace(0, T_max, 30))
plt.yscale("log")
plt.xlabel("IRT value")
plt.ylabel("# count")

plt.subplot(133)
df_fanout = df.groupby(["N_recalled", "position"]).agg({"irts": np.mean})
df_fanout = pd.pivot_table(
    data=df, index=["N_recalled", "position"], values="irts", aggfunc=np.nanmean
).reset_index()
for i_N in range(16):
    data2plot = df_fanout[df_fanout["N_recalled"] == i_N]
    plt.plot(data2plot["irts"].values)

plt.xlabel("Transition number")
plt.ylabel("IRT value")

# plt.subplot(224)
# plt.plot(rates_np[0][:,range(0,20,1)].T)
# plt.xlabel('time')
# plt.ylabel('rate')

plt.savefig(filename + "_fig3.svg")
plt.savefig(filename + "_fig3.pdf")
# plt.show()
plt.close("all")

#%% Figure 4
N_bins = 1000
sizes = similarities[:, range(16), range(16)]
sizesretrieved = df["sizes"][df["sizes"] > 0]
bins = np.percentile(sizesretrieved, np.linspace(0, 100, N_bins + 1))
sizes_bin = np.digitize(sizes, bins)
sizesretrieved_bin = np.digitize(sizesretrieved, bins)
sizes_counts = [np.sum(sizes_bin == x) for x in range(N_bins)]
sizesretrieved_counts = [np.sum(sizesretrieved_bin == x) for x in range(N_bins)]
bins_centers = (bins[:-1] + bins[1:]) / 2
values_prob = np.array(sizesretrieved_counts) / np.array(sizes_counts)
plt.scatter(bins_centers, values_prob)
plt.savefig(filename + "_fig4.svg")
plt.savefig(filename + "_fig4.pdf")
# plt.show()
plt.close("all")

#%% Figure 5

plt.subplot(121)
values = df["ranks"][df["ranks"] > 0]
counts = [np.sum(values == x) for x in range(1, 15)]
plt.plot(range(1, 15), counts / np.sum(counts))
plt.xlabel("transition ranks")
plt.ylabel("probability distribution")

plt.subplot(122)
N_bins = 50
intersections = df["intersections"][df["intersections"] > 0]
irts = df["irts"][df["intersections"] > 0]
bins = np.percentile(intersections, np.linspace(0, 100, N_bins + 1))
intersections_bin = np.digitize(intersections, bins)
irts_mean = [np.mean(irts[intersections_bin == x]) for x in range(N_bins)]
bins_centers = (bins[:-1] + bins[1:]) / 2
plt.scatter(bins_centers, irts_mean)
plt.xlabel("memories similarity")
plt.ylabel("average IRT")
plt.savefig(filename + "_fig5.svg")
plt.savefig(filename + "_fig5.pdf")
# plt.show()
plt.close("all")


#%%
