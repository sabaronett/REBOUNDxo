import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# load memory output data
data = np.loadtxt('mem.txt') # return (N, 2) array
step = data[:, 0]                   # return only 1st col
mem = data[:, 1]                    # return only 2nd col

fig, (ax1) = plt.subplots(nrows=1, ncols=1)
ax1.set_xlabel('Process Time / s', fontsize='large')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.set_ylabel('Memory Usage / MB', fontsize='large')
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.plot(step, mem)
ax1.grid()
plt.show()
# plt.savefig('plot/mem.png')
