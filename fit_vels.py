import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_interp import *
from f_fits import fit_vel

if len(sys.argv) != 5:
    print("Usage: python3 fit_vels.py ratio subdir arm dt")
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

R, vm, v_std = monte_carlo_velocities(all, sigma_R=2.5, dt=dt, n_mc=1000)

#compute real unperturbed velocity
interp_data = int_all(all, n=270, i=0)
smooth_data = smooth(interp_data, 1)
vel = compute_velocity(smooth_data, dt)
vm = np.array([v[:,1] for v in vel])  # drop the R coord

plot_R_data_sets(R, vm, "v", dt)

#fits
for it in range(vm.shape[0]):
    v_obs = vm[it]
    v_err = v_std[it]

    rs1, chi1, chir1, p1, A = fit_vel(R, v_obs, v_err, vkep)
    rs2, chi2, chir2, p2, B = fit_vel(R, v_obs, v_err, vcost)

    print("time ", dt*it, "-", dt*(it+1))
    print("r quad: ", rs1, " ", rs2)
    print("chi quad: ", chi1, " ", chi2)
    print("chi red: ", chir1, " ", chir2)
    print("p value: ", p1, " ", p2)

    plt.plot(R, vkep(R, A), label="keplerian fit")
    plt.plot(R, vcost(R, B), label="COnstant fit")

plt.show()