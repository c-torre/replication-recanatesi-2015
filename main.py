# %%
import os
import pickle  # to be replaces with .npy files

from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.style as mpstyle
import numpy as np

RESULTS_DIR = "results"
# To use when refactoring without pickle dictionary
# ACTIVITIES_DIR = os.path.join(RESULTS_DIR, "activities")
# SIMILARITIES_DIR = os.path.join(RESULTS_DIR, "similarities")


try:
    job_env = int(os.getenv("SLURM_ARRAY_TASK_ID")) # Changes seed per cluster simulation
except:
    job_env = 0 # Default seed for non-cluster use
np.random.seed(job_env)

mpstyle.use("fast")  # Faster plots for prototyping

NUM_NEURONS = 10 ** 5  # "N"
NUM_MEMORIES = 16  # "P"
# Activation
T_DECAY = 0.01  # "tau"
# Time
T_STEP = 0.001  # "dt"
T_TOT = 450 # "T"; Half time iterations for prototyping (default: 450)
T_SIMULATED = int(T_TOT // T_STEP)
# Hebbian rule
EXCITATION = 12_500  # "J1"
SPARSITY = 0.1  # "f0"
# Gain
GAIN_THRESHOLD = 0  # set as *(x > 0) in gain function in matlab
GAIN_EXP = 1 / 3  # set in gain function in matlab
# Inhibition
SIN_MIN = 0.4 * EXCITATION  # "J_0m"
SIN_MAX = 1.2 * EXCITATION  # "J_0M"
# Noise
NOISE_STD = 65  # "noise_amp"
# Forward and backward contiguity
CONT_FORTH = 1500 / NUM_NEURONS  # "J_f"
CONT_BACK = 850 / NUM_NEURONS  # "J_b"
# R
NUM_POINTS = T_TOT * 100  # ?? from matlab script

def check_state(state):
    STATES = ["SEED", "CONT_FORTH", "NOISE"]
    assert state in STATES

state = "CONT_FORTH"
check_state(state)

cont_forth = seed / NUM_POINTS
cont_back = seed / NUM_POINTS

# #%%
def build_logical(prev_logical):
    composite = np.hstack((prev_logical, prev_logical))
    try:
        half_size = prev_logical.shape[1]
    except:
        half_size = 1
    bottom = np.hstack((np.zeros(half_size), np.ones(half_size)))
    composite = np.vstack((composite, bottom))
    return composite


def get_log_int_sizes():
    population_sizes = np.array([NUM_NEURONS])
    log_int = np.array(0)

    for i in range(NUM_MEMORIES):
        new_sizes = np.array(
            [
                np.random.binomial(population_sizes[x], SPARSITY)
                for x in np.arange(population_sizes.shape[0])
            ]
        )
        population_sizes = np.hstack(((population_sizes - new_sizes), new_sizes))
        log_int = build_logical(log_int)

    log_int = log_int[:, population_sizes > 0]
    population_sizes = population_sizes[population_sizes > 0]

    return population_sizes, log_int


def get_connectivities(log_int):
    connectivity_reg = log_int.T
    # connectivity_reg = (log_int)[1:].T
    connectivity_back = np.hstack(
        (
            np.zeros(connectivity_reg.shape[0])[:, None],
            connectivity_reg[:, : NUM_MEMORIES - 1],
        )
    )
    connectivity_forth = np.hstack(
        (connectivity_reg[:, 1:], np.zeros(connectivity_reg.shape[0])[:, None])
    )
    return connectivity_reg, connectivity_back, connectivity_forth


def prepare_times():
    it_rec = np.linspace(1, T_SIMULATED, NUM_POINTS).astype(int)
    time = np.arange(start=0, stop=T_TOT + T_STEP, step=T_STEP)
    t_rates = time[it_rec - 1] - 1
    return it_rec, time, t_rates


def osc(t):
    return (SIN_MIN + SIN_MAX) / 2 + (SIN_MIN - SIN_MAX) / 2 * np.sin(
        2 * np.pi * t + np.pi / 2
    )


def noise(noise_std, population_sizes):
    return (
        noise_std
        / np.sqrt(population_sizes)
        * np.random.randn(population_sizes.shape[0])
        / np.sqrt(T_STEP)
    )


def gain(currents_vector):
    # return (np.heaviside(currents_vector, 0) * currents_vector) ** 0.4
    # def func(x):
    # print("GAIN")
    # currents_vector += np.random.choice([-1, 1]) * 10 ** -8  # Division by 0
    # print(currents_vector)
    adaption = np.heaviside(currents_vector, 0)
    # print(adaption)
    currents_vector *= adaption
    #gains = np.exp(-1 / currents_vector ** 2) * (currents_vector ** GAIN_EXP)
    gains = currents_vector ** GAIN_EXP

    # gains *= (np.exp(-1 / currents_vector ** 2) * (currents_vector ** GAIN_EXP)).astype(
    #     float
    # )
    # print(gains)
    # print("GAIN")
    return gains
    # return (np.heavisi


def get_initial_currents(num_memories, connectivity_reg):
    rnd_memory = np.ceil(np.random.random(10) * num_memories) - 1
    # rnd_memory = np.random.randint(1, num_memories)
    currents_init = connectivity_reg[:, rnd_memory]
    # currents_init += connectivity_reg[:, 0]
    return currents_init


def mult_W(Sgs, Vs, connectivity_reg, connectivity_back, connectivity_forth):
    sparsity_vect = np.ones(NUM_MEMORIES) * SPARSITY
    result = (
        connectivity_reg @ Vs  # (65536,) @ (16,) = (65536,)
        - sparsity_vect @ Vs  # (16,) @ (16,) = (1,)
        - connectivity_reg
        @ sparsity_vect
        * np.sum(Sgs)  # (65536, 16) @ (16,) * (1,) = (65536,)
        + sparsity_vect @ sparsity_vect * np.sum(Sgs)  # (16,) @ (16,) * (1,) = (1,)
        + CONT_FORTH * connectivity_forth @ Vs  # (1,) * (65536, 16) @ (16,) = (65536,)
        + CONT_BACK * connectivity_back @ Vs  # (1,) * (65536, 16) @ (16,) = (65536,)
    )
    return result


#%
def evolv(
    t, curr, population_sizes, connectivity_reg, connectivity_back, connectivity_forth,
):
    act = gain(curr)
    sized_activations = population_sizes * act
    Vs = connectivity_reg.T @ sized_activations
    mult_W_ = mult_W(
        sized_activations, Vs, connectivity_reg, connectivity_back, connectivity_forth,
    )
    noise_ = NOISE_STD / np.sqrt(population_sizes) *np.random.randn(population_sizes.shape[0]) / np.sqrt(T_STEP)
    # noise_ = noise(NOISE_STD, population_sizes)
    sine = osc(t)
    result = (1 / T_DECAY * (-1 * curr + 1 / NUM_NEURONS * (EXCITATION * mult_W_ - sine * np.sum(population_sizes * act)) + noise_))
    # print(result)
    return result



#%% THE SCRIPT STARTS HERE
N = NUM_NEURONS
P = NUM_MEMORIES
f = SPARSITY
Xi = (np.random.random((P,N)) < f).T

V = np.array([[int(i) for i in bin(num)[2:].zfill(P)] for num in range(2**P)])
powers_of_two = np.array([2 ** i for i in range(P-1,-1,-1)])
v_n = Xi @ powers_of_two
S = np.histogram(v_n, np.arange(-1,2**P)+0.5)[0]
V = V[S>0,:]
S = S[S>0]

population_sizes_ = S
log_int_ = V.T
# population_sizes_, log_int_ = get_log_int_sizes()

num_pops = population_sizes_.shape[0]
connectivity_reg_, connectivity_back_, connectivity_forth_ = get_connectivities(log_int_)
it_rec, time, t_rates = prepare_times()
rates = np.zeros((num_pops, len(it_rec)))
ind_t = 0

rnd_memory = (np.ceil(np.random.random(1) * P) - 1).astype(int)
currents = np.squeeze(connectivity_reg_[:, rnd_memory].astype(np.float))

sparsity_vect = np.ones(NUM_MEMORIES) * SPARSITY
rates = np.zeros((num_pops, len(time)))
for it in range(len(time)):
    t = time[it]
    act = gain(currents.copy())
    sized_activations = population_sizes_ * act
    r_tot = np.sum(sized_activations)
    noise_ = NOISE_STD / np.sqrt(population_sizes_) * norm.ppf(np.random.rand(population_sizes_.shape[0]))
    Vs = sized_activations @ connectivity_reg_
    mult_W_ =  connectivity_reg_ @ Vs- sparsity_vect @ Vs- connectivity_reg_ @ sparsity_vect * r_tot + sparsity_vect @ sparsity_vect * r_tot
    contiguity_ = CONT_FORTH * connectivity_forth_ @ Vs + CONT_BACK * connectivity_back_ @ Vs
    sine = osc(t) / N
    evolv_ = T_STEP / T_DECAY * (-currents + EXCITATION / NUM_NEURONS * (mult_W_ + contiguity_) - sine * r_tot + noise_ / np.sqrt(T_STEP))
    currents += evolv_
    rates[:, it] = act

#%

# currents, rates, connectivity_reg_, population_sizes_ = main()

# pickle.dump(rates, open(os.path.join(".", "rates.p"), "wb"))
# rates = pickle.load(open(os.path.join(".", "rates.p"), "rb"))

######### Most plotting stuff commented out for cluster
# #%%
proj_attr = (connectivity_reg_ * population_sizes_[:, None]).T
similarity = proj_attr @ connectivity_reg_
rate_avg = (S * rates.T @ connectivity_reg_  / np.diagonal(similarity)).T
#%
to_save_dict = {"rate_avg": rate_avg, "similarity": similarity}
# np.save(os.path.join(ACTIVITIES_DIR, f"s{job_env}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{NOISE_STD}-activities"), rate_avg)
# np.save(os.path.join(SIMILARITIES_DIR, f"s{job_env}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{NOISE_STD}-similarities"), similarity)
#%
file_path = os.path.join(
    RESULTS_DIR, f"s{job_env}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{NOISE_STD}.p"
)  ############ MEEEEEEEEEEEEEEEEEEEEEE
pickle.dump(to_save_dict, open(file_path, "wb"))  ######## MEEEEEEEEEEEEEEEEEE
# np.save(os.path.join(RESULTS_DIR, f"s{job_env}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{NOISE_STD}"), to_save_dict)
# row, col = np.where(rate_avg > 15)
#########import seaborn as sns
#########plt.subplot(211)
#########to_line_plot = rate_avg.T  # pd.DataFrame(np.rot90(rate_avg[:, :100])) commented for cluster
#########fig_activities = sns.lineplot(data=to_line_plot, dashes=False, palette="colorblind") #commented for cluster
########## plt.show() commented for cluster
#########
#########plt.subplot(212)
#########fig_attractors = sns.heatmap(rate_avg) #commented for cluster
#########plt.show()
##########%%
