#%%
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
from IPython.display import display, clear_output


class Model:
    def __init__(self):
        self.GAIN_THRESHOLD = 0
        self.GAIN_EXPONENT = 0.5
        self.NUM_NEURONS = 10 ** 5
        self.NUM_MEMORIES = 16
        self.SPARSITY = 0.01
        self.EXCITATION = 11
        self.NOISE_VAR = 1
        self.T_STEP = 1e-2
        self.T = 10
        self.OSCILLATION = 0
        self.T_DECAY = 1e-1

    def make_patterns(self):
        self.patterns = scipy.sparse.random(
            self.NUM_NEURONS,
            self.NUM_MEMORIES,
            density=self.SPARSITY,
            data_rvs=lambda n: np.ones((n,)),  # makes sparsity definition as in paper
        )

    def gain(self, x):
        # print(x)
        x1 = np.where((self.GAIN_THRESHOLD + x) <= 0.0, 0, self.GAIN_THRESHOLD + x)
        return np.power(x1, self.GAIN_EXPONENT)

    def Jr(self, currents):
        activations = self.gain(currents)
        p_currents = (
            self.patterns.T.dot(activations) - self.SPARSITY * activations.sum()
        )
        return (
            (
                self.patterns.dot(p_currents)
                - self.SPARSITY * p_currents.sum()
                - activations * self.OSCILLATION
            )
            * self.EXCITATION
            / self.NUM_NEURONS
        )

    def grad(self, currents):
        return (
            -currents
            + self.Jr(currents)
            + np.random.randn(currents.shape[0], currents.shape[1]) * self.NOISE_VAR
        ) / self.T_DECAY


#%%

np.random.seed(12354)
mod = Model()
mod.NUM_NEURONS = 10 ** 4
mod.NUM_MEMORIES = 16
mod.SPARSITY = 0.1
# mod.patterns*=0
mod.NOISE_VAR = 0.5
mod.EXCITATION = 1000

phi_amp = 400
phi_0 = phi_amp + 600

T_CYCLES = 1  # 500
T_TOT = int(T_CYCLES / mod.T_STEP)

#%%

# for it in tqdm(range(1000)):
for sim in (range(1)):
    # sys.stdout.write(str(it) + "        \r")
    mod.make_patterns()
    currents_all = np.empty((mod.NUM_NEURONS, T_TOT))
    currents = np.random.randn(mod.NUM_NEURONS, 1)
    currents = (mod.patterns.tocsr()[:, 0]).todense()
    currents_all[:, :1] = currents
    # break  ################XXX
    for t_step in tqdm(range(1, currents_all.shape[1])):
        currents += mod.grad(currents) * mod.T_STEP
        currents_all[:, t_step] = np.array(currents)[:, 0]
        mod.phi = phi_0 - phi_amp * np.sin(t_step / 50)

#%%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
display(fig)

mem_rec = np.zeros((T_TOT,))
t_rec_all = []
ITR = np.empty((0, 1))
itrByLen = [np.empty((k - 1, 0)) for k in range(1, mod.NUM_MEMORIES + 1)]

#%%

mu = (
    mod.patterns.T.dot(mod.gain(currents_all))
    - mod.SPARSITY * mod.gain(currents_all).sum(axis=0)
).T / mod.NUM_NEURONS
#%%
mem1 = ((mu > 0.5).cumsum(axis=0) > 0).sum(axis=1)
#%%
trec = np.argwhere(np.diff(mem1))
#%%
t_rec_all.append(trec)
ITR = np.vstack((ITR, np.diff(trec, axis=0)))
#%%
mem_rec += mem1
#%%
itrByLen[len(trec) - 1] = np.hstack((itrByLen[len(trec) - 1], np.diff(trec, axis=0)))
ax1.cla()
#%%
ax1.plot(mem_rec / len(t_rec_all))
#%%
ax2.cla()
#%%
ax2.hist(ITR, bins=100)
#%%
ax2.set_xlim([0, 10000])
#%%
ax2.set_yscale("log")
#%%
ax3.cla()
#%%
for itr in itrByLen:
    if itr.size > 0:
        ax3.plot(itr.mean(axis=1))
clear_output(wait=True)
#%%
display(fig)
fig.savefig("res.png", dpi=600)


# %%
