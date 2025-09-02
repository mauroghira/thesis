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
    print("Usage: python interpoling.py ratio subdir arm dt")
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

#plot_all_phi_r(all, "data", dt)

interp_data = int_all(all, n=270, i=0)
#plot_all_phi_r(interp_data, "interp", dt)

smooth_data = smooth(interp_data, 1)
plot_all_phi_r(smooth_data, "smooth", dt)

vel = compute_velocity(smooth_data, dt)

plot_vel(vel, dt)

if subdir == "sim_ana":
    file = base + ratio + "/results/sim_" + arm + "_int_"+str(dt)+"_phi.txt"
else:
    file = base + ratio + "/results/mc_" + arm + "_int_"+str(dt)+"_phi.txt"
save_rphi_all(smooth_data, file)

if subdir == "sim_ana":
    file = base + ratio + "/results/sim_" + arm + "_int_"+str(dt)+"_vel.txt"
else:
    file = base + ratio + "/results/mc_" + arm + "_int_"+str(dt)+"_vel.txt"
save_vel(vel, file)




#all[1] = extrapolate_phi_in(all[1], 30, 48,30,53)
#all[2] = extrapolate_phi_in(all[2], 50, 75, 50)
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