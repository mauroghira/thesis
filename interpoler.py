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

#sm_r_data = smooth(all_data, 0)
sm_r_data = sort_and_monotonic_smooth(all_data)

#plot_all_r_phi(sm_r_data, "Smoothed data")
#plot_all_r_phi(all_data, "original data")
#plt.show()

interp_data = int_all(all_data, n=100)

#smooth_data = smooth(interp_data, 1)

#plt.plot(sm_r_data[0][:, 1], sm_r_data[0][:, 0], label=f'Smoothed')
#plt.plot(interp_data[0][:, 0], interp_data[0][:, 1], label=f'Interpolated')
#plot_all_phi_r(smooth_data, "Smoothed data")
plot_all_phi_r(interp_data, "Interpolated data")
#plt.show()

velocities = compute_velocity(interp_data, dt=1)
#save_vel(velocities)

plot_vel(velocities)