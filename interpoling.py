import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_interp import *

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
    if os.path.exists(input_1):
        data = read_single(input_1)

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