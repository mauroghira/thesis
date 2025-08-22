import numpy as np
import sys
import matplotlib.pyplot as plt
from fuctions import *
from other_fun import *

if len(sys.argv) != 3:
    print("Usage: python interpoler.py directory arm")
    sys.exit(1)

dir = sys.argv[1]
arm = sys.argv[2]

all_data = read_mult(dir, arm)
interp_data = int_all(all_data)

smooth_data = smooth(interp_data)

velocities = compute_velocity(smooth_data)

plot_vel(velocities)

#plt.plot(smooth_data[0][:, 0], smooth_data[0][:, 1], label=f'Smoothed')
#plt.plot(interp_data[0][:, 0], interp_data[0][:, 1], label=f'Interpolated')
#plot_all(smooth_datam, "Smoothed data")
#plot_all(interp_data, "Interpolated data")
#plt.show()