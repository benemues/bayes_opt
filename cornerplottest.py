import corner
import numpy as np
import matplotlib.pyplot as plt

ndim, nsamples = 2, 10000
np.random.seed(42)
samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
figure = corner.corner(samples)
plt.show()