import numpy as np
from utils import nearestPD, isPD

if __name__ == '__main__':
    for i in range(10):
        for j in range(2, 100):
            A = np.random.randn(j, j)
            B = nearestPD(A)
            assert(isPD(B))
    print('unit test passed!')