import build.Release.TEST_PYBIND_CUDA as E
import numpy as np

size = int(1e6)
a = np.ones(size)
b = 2 * np.ones(size)

print(E.add_cuda(a, b))