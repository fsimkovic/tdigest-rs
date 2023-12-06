import time

import numpy as np
from joblib import Parallel, delayed

from tdigest_rs import TDigest

quantile = 0.1
n = 16_000
n_arrays = 5000

arrays = [np.random.randn(n).astype(np.float32) for _ in range(n_arrays)]

t0 = time.time()
tdigests = Parallel(backend="threading", verbose=3, n_jobs=-1)(
    delayed(TDigest.create)(arr=arr, delta=10.0, sort_in_place=True) for arr in arrays
)
print(f"Total running time parallel: {time.time() - t0}")

# t0 = time.time()
# for arr in arrays:
#     tdigest = TDigest.create(arr=arr, delta=10.0)
#     # q = tdigest.quantile(0.5)

# print(f"Total running time: {time.time() - t0}")
