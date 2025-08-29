import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_post_processing import *

if len(sys.argv) != 3:
    print("Usage: python fitting.py directory arm")
    sys.exit(1)

dir = sys.argv[1]
arm = sys.argv[2]

N = 100
dt = 5

all = []
for i in range(0, 11, dt):
    input_1 = dir + str(i) + "_" + arm + ".txt"
    input_2 = dir + str(i) + "_i" + arm + ".txt"
    if os.path.exists(input_1):
        data = read_single(input_1)
    if os.path.exists(input_2):
        data2 = read_single(input_2)
        data = np.vstack((data2, data))

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

save_rphi_all(interp_data)
save_vel(vel)
