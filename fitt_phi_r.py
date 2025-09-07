import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_fits import *

if len(sys.argv) != 5:
    print("Usage: python fitting.py ratio subdir arm dt")
    sys.exit(1)

base = os.path.expanduser("~/thesis/Spiral_pattern/")
ratio = sys.argv[1]
subdir = sys.argv[2]
arm = sys.argv[3]
dt = int(sys.argv[4])

N = 100

all = []
for i in range(0, 11, dt):
    input_1 = base + ratio + "/" + subdir + "/" + str(i) + "_" + arm + ".txt"
    #input_2 = base + ratio + "/" + subdir + "/" + str(i) + "_i" + arm + ".txt"
    if os.path.exists(input_1):
        data = read_single(input_1)
    """
    if os.path.exists(input_2):
        data2 = read_single(input_2)
        data = np.vstack((data2, data))
    """
    all.append(data)

pm, PM = select_extremes(all, 1)
rm, RM = select_extremes(all, 0)

interp_data = []
for data in all:
    r = data[:,0]
    phi = data[:,1]

    candidates = [
        ("fitted r(phi) with log",     spiral,    (phi, r, "r")),
        ("fitted r(phi) with arch",    archspir,  (phi, r, "r")),
        ("fitted phi(r) with log",     logspir,   (r, phi, "phi")),
        ("fitted phi(r) with arch",    archspir,   (r, phi, "phi")),
    ]
    popt, mode, model = fit_models(r,phi, candidates)
    if popt is None:
        continue

    if mode == "phi":
        r_fit = np.linspace(rm, RM, N)
        phi_fit = model(r_fit, *popt)

    else:
        r_fit, phi_fit = resample_rphi(model, popt, rm, RM, (pm,PM), N)
    
    interp_data.append(np.column_stack((r_fit,phi_fit)))

plot_all_phi_r(interp_data, "data", dt)

vel = compute_velocity(interp_data, dt)
plot_vel(vel, dt)


if subdir == "sim_ana":
    file = base + ratio + "/results/sim_" + arm + "_fit_"+str(dt)+"_phi.txt"
else:
    file = base + ratio + "/results/mc_" + arm + "_fit_"+str(dt)+"_phi.txt"
save_rphi_all(interp_data, file)

if subdir == "sim_ana":
    file = base + ratio + "/results/sim_" + arm + "_fit_"+str(dt)+"_vel.txt"
else:
    file = base + ratio + "/results/mc_" + arm + "_fit_"+str(dt)+"_vel.txt"
save_vel(vel, file)