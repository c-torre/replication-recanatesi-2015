import numpy as np
import matplotlib.pyplot as plt

seed = np.random.randint(0, 2 ** 10)
print("seed", seed)
np.random.seed(seed)

xi_0 = 65
n_neurons = 1000
n_iteration = 2000

std = xi_0 ** 0.5

data_single = np.sum(
    np.random.normal(loc=0, scale=std,
                     size=(n_neurons, n_iteration)), axis=0)

data_pop = np.random.normal(
    loc=0, scale=std * n_neurons ** 0.5, size=n_iteration)

fig, axes = plt.subplots(nrows=2, figsize=(10, 4))

line_width = 0.5

data = data_single, data_pop
titles = "Single unit", "Population"

for i in range(len(data)):

    ax = axes[i]
    ax.plot(data[i], linewidth=line_width)
    ax.set_title(titles[i])
    ax.set_xlim(0, len(data[i]))

plt.show()