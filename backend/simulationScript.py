import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import os
import sys

Width = sys.argv[1]
Depth = sys.argv[2]
Temperature = sys.argv[3]
Pressure = sys.argv[4]
Timestep = sys.argv[5]

# Insert numerical shabang here!


# Test plot
yoyoyo = np.arange(0, 10, 0.1)
yoyo = np.sin(yoyoyo)

plt.plot(yoyoyo, yoyo)
plt.savefig('./images/figure.png')

print("OK")
