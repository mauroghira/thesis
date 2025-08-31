import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_post_processing import *

if len(sys.argv) != 5:
    print("Usage: python join_vel.py ratio sbudir arm dt")
    sys.exit(1)

base = os.path.expanduser("~/thesis/Spiral_pattern/")
ratio = sys.argv[1]
subdir = sys.argv[2]
arm = sys.argv[3]
dt = int(sys.argv[4])

all = []
for i in range(0, 11, dt):
    input_1 = base + ratio + "/" + subdir + "/" + str(i) + "_" + arm + ".txt"
    input_2 = base + ratio + "/" + subdir + "/" + str(i) + "_i" + arm + ".txt"
    if os.path.exists(input_1):
        data = read_single(input_1)
    if os.path.exists(input_2):
        data2 = read_single(input_2)
        data = np.vstack((data2, data))

    all.append(data)

all[1] = extrapolate_phi_in(all[1], 30, 48,30,53)
#all[2] = extrapolate_phi_in(all[2], 50, 75, 50)

plot_all_phi_r(all, "data", dt)

"""
for i, data in enumerate(all):
    #data = data[data[:, 0] >= 40]
    #data = fill_phi_gaps(data)
    #data = filter_bads(data, -10, 30)
    #data = filter_bads(data, 100, 135)
    #data = extrapolate_phi_in(data, 40,50)

    if i*dt == 10:
        #data = extrapolate_phi_in(data, 77, 100)
        #data = extrapolate_phi_in(data, 45, 60)
        data = extrapolate_phi_in(data, 30, 50)

    if i*dt == 0:
        data = extrapolate_phi_in(data, 30, 50)

    elif i*dt == 5:
        data = extrapolate_phi_in(data, 20, 23)
    
    #int_data = interp(data, np.min(data[:,0]), np.max(data[:,0]), 200, 0)

    all[i] = data
#"""

interp_data = int_all(all, n=270, i=0)
plot_all_phi_r(interp_data, "data", dt)

smooth_data = smooth(interp_data, 1)
#plot_all_phi_r(smooth_data, "data", dt)

vel = compute_velocity(smooth_data, dt)

plot_vel(vel, dt)

name = input("filename> ")
file = base + ratio + "/results/" + name
save_rphi_all(smooth_data, file)

name = input("filename> ")
file = base + ratio + "/results/" + name
save_vel(vel, file)