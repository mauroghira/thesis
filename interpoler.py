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

dt = 10

all_data = read_mult(dir, arm, dt)
interp_R = []

for i, data in enumerate(all_data):
    """per estermo
    data = fill_phi_gaps(data)
    data = filter_bads(data, -10, 30)
    if i*dt == 10:
        data = extrapolate_phi_in(data, 77, 100)
    
    """#per interno
    if i*dt == 0:
        data = extrapolate_phi_out(data, 42)
    int_data = interp(data, np.min(data[:,0]), np.max(data[:,0]), 100, 0)
    interp_R.append(int_data)
    #"""
    all_data[i] = data

    #data = fill_phi_gaps(data)
    #data = filter_bads(data, 90, 135)
    #data = smooth_int(data, 20, 28)

#plot_all_phi_r(all_data, "data")

#interp_R = int_all(all_data, n=100, i=1)

#smooth_data = smooth(interp_R, 0)
interp_data = int_all(interp_R, n=100, i=0)
smooth_data = smooth(interp_data, 1)

plot_all_phi_r(smooth_data, "Smoothed data")
plot_all_phi_r(interp_data, "Interpolated data")

int_vel = compute_velocity(smooth_data, dt)

plot_vel(int_vel)