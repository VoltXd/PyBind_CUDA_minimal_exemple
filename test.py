import build.Release.TEST_PYBIND_CUDA as E
import numpy as np
import matplotlib.pyplot as plt

size = int(1e6)

param = [[0, 1], [3, 2], [-5, 0.5]]
legend_str = []

colors = ["blue", "red", "green"]


for i in range(len(param)):
    mu = param[i][0]
    sigma = param[i][1]

    a = E.get_random_gauss_vector(mu, sigma, size)

    num_bins = 50

    n, bins, patches = plt.hist(a, num_bins, density=1, color=colors[i], alpha=0.7)

    y = ((1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    plt.plot(bins, y, '--', color ="black")
    
    plt.xlabel('X')
    plt.ylabel('Density')

    legend_str.append("Normal probability density")

for i in range(len(param)):
    legend_str.append(r"Random numbers histogram $(\mu = {}, \sigma = {})$".format(param[i][0], param[i][1]))

plt.grid()
plt.legend(legend_str)
plt.show()