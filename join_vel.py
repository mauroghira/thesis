import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from fuctions import *
from other_fun import *

if len(sys.argv) != 3:
    print("Usage: python join_vel.py directory arm")
    sys.exit(1)

dir = sys.argv[1]
arm = sys.argv[2]

dt = 5

all = []
for i in range(0, 11, dt):
    input_1 = dir + str(i) + "_" + arm + ".txt"
    input_2 = dir + str(i) + "_i" + arm + ".txt"
    if os.path.exists(input_1):
        data1 = read_single(input_1)
    if os.path.exists(input_2):
        data2 = read_single(input_2)
    data = np.vstack((data2, data1))

    all.append(data)

#plot_all_phi_r(all, "data")

for i, data in enumerate(all):
    data = fill_phi_gaps(data)
    data = filter_bads(data, -10, 30)
    data = filter_bads(data, 100, 135)
    data = extrapolate_phi_in(data, 42, 50)
    if i*dt == 10:
        data = extrapolate_phi_in(data, 77, 100)
    if i*dt == 5:
        data = extrapolate_phi_in(data, 20, 23)
    int_data = interp(data, np.min(data[:,0]), np.max(data[:,0]), 200, 0)

    all[i] = int_data

interp_data = int_all(all, n=100, i=0)
smooth_data = smooth(interp_data, 1)
vel = compute_velocity(smooth_data, dt)

plot_all_phi_r(smooth_data, "data", dt)
plot_vel(vel, dt)