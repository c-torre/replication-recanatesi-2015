# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt

# from tqdm.notebook import tqdm
from tqdm import tqdm

job_env = int(os.getenv("SLURM_ARRAY_TASK_ID"))  # CHANGE MEEEEEEEEEEEEEEEEE
#job_env  = 1
np.random.seed(job_env)

import sys
from IPython.display import display, clear_output
class Model:
    def __init__(self):
        self.th0 = 0
        self.gamma = 0.5
        self.N = 100000
        self.P = 16
        self.f = 0.01
        self.kappa = 11
        self.xi0 = 1
        self.dt = 1e-2
        self.T = 10
        self.phi = 0
        self.tau = 1e-1

    def makePatterns(self):
        self.eta = scipy.sparse.random(self.N, self.P, density=self.f,
                                       data_rvs=lambda n: np.ones((n,)))

    def gain(self, x):
        x1 = np.where((self.th0 + x) <= 0., 0, self.th0 + x)
        return np.power(x1, self.gamma)

    def Jr(self, c):
        r = self.gain(c)
        pc = self.eta.T.dot(r) - self.f * r.sum()
        return (self.eta.dot(
            pc) - self.f * pc.sum() - r * self.phi) * self.kappa / self.N

    def grad(self, c):
        return (-c + self.Jr(c) + np.random.randn(c.shape[0], c.shape[
            1]) * self.xi0) / self.tau

np.random.seed(12354)
mod = Model()
mod.N = 10000
mod.P = 16
mod.f = 0.1
# mod.eta*=0
mod.xi0 = 0.5
mod.kappa = 1000

phiAmp = 400
phi0 = phiAmp + 600
nT = 25000#00 MY CHANGERINO FROM 50000
memRec = np.zeros((nT,))
tRecAll = []
ITR = np.empty((0, 1))
itrByLen = [np.empty((k - 1, 0)) for k in range(1, mod.P + 1)]

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
#                                              figsize=(12, 5))
# ax4.set_axis_off()
# display(fig)
num_simulations = 1  # I ADDERINO AND MAYBE CHANGERINO FROM 1000

# for sim in (range(num_simulations)):
mod.makePatterns()
currents_all = np.empty((mod.N, nT))
# c = np.random.randn(mod.N, 1)
currents = (mod.eta.tocsr()[:, 0]).todense()
np.save(f"results/patterns/s{job_env}-n{mod.xi0}-patterns", mod.eta.tocsr().todense())
#%%
currents_all[:, :1] = currents
for t_step in tqdm(range(1, currents_all.shape[1])):
    currents += mod.grad(currents) * mod.dt
    currents_all[:, t_step] = np.array(currents)[:, 0]
    mod.phi = phi0 - phiAmp * np.sin(t_step / 50)
# mu is (10000, 16)
mu = (mod.eta.T.dot(mod.gain(currents_all)) - mod.f * mod.gain(currents_all).sum(
    axis=0)).T / mod.N
"""
Double masking:
First selects some current modification above 0.5
That makes 0 all False for the cumsum() in the vertical axis
Then it masks again for those above 0 and makes horizontal sum
"""
mem1 = ((mu > 0.5).cumsum(axis=0) > 0).sum(axis=1)
np.save(f"results/recalls/s{job_env}-n{mod.xi0}-recalls", mem1)
"""
take where there is no zero of the diffs (num minus previous)
"""
trec = np.argwhere(np.diff(mem1))
# the list is updated with an array, and one added every simulation
tRecAll.append(trec)
ITR = np.vstack((ITR, np.diff(trec, axis=0)))
memRec += mem1
itrByLen[len(trec) - 1] = np.hstack(
    (itrByLen[len(trec) - 1], np.diff(trec, axis=0)))
# ax1.cla()
# ax1.plot(memRec / len(tRecAll))
# ax2.cla()
# ax2.hist(ITR, bins=100)
# ax2.set_xlim([0, 10000])
# ax2.set_yscale('log')
# ax3.cla()
# for itr in itrByLen:
#     if itr.size > 0:
#         ax3.plot(itr.mean(axis=1))
# clear_output(wait=True)
# display(fig)
# fig.savefig('res.png', dpi=600)
