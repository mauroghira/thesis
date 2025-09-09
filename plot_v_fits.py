import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_interp import *
from f_fits import *
from f_read import *

titlefontsize, labelfontsize, tickfontsize, legfontsize = 20, 16, 14, 12
markersize, linewidth = 6, 1

if len(sys.argv) != 3:
    print("Usage: python plot_v_fits.py ratio arm")
    sys.exit(1)

dt = 10
ratio = sys.argv[1]
arm = sys.argv[2]
zero, rr = ratio.rsplit("0", 1)

outfile = "~/thesis/img_ts/"+ratio+"_vel_fits"
base = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/")
fine = "_int_"+str(dt)+"_vel.txt"

#add data from simulations
if arm == "top":
    file = base+"sim_top"+fine
else:
    file = base+"sim_bot"+fine
Rs, vms, v_stds = load_vel(file)

#add data from observations
if arm == "top":
    file = base+"mc_top"+fine
else:
    file = base+"mc_bot"+fine
Ro, vmo, v_stdo = load_vel(file)


fig, axs = plt.subplots(1, 2, figsize=(17, 9))
title = 'q=0.'+str(rr)+fr', dt=10 years, '+arm+' arm'

#èòot simulation datta
ax = axs[0]
R = Rs
v_sim = np.asarray(vms).flatten()
v_err = np.asarray(v_stds).flatten()

rs1, chi1, chir1, p1, A, devA = fit_vel(R, v_sim, v_err, vkep)
rs2, chi2, chir2, p2, B, devB = fit_vel(R, v_sim, v_err, vcost)

# line for the velocities
ax.plot(R, v_sim, '-', linewidth=linewidth,
        label=f"Measured speed")
# shaded error region
ax.fill_between(R, v_sim - v_err, v_sim + v_err, alpha=0.2, color=ax.lines[-1].get_color(), label="Error bar")
ax.plot(R, vkep(R, A), '-.', label=fr'Keplerian fit - $\chi^2$ = {chir1:.3f}', linewidth=linewidth)
ax.plot(R, vcost(R, B), '--', label=fr'Constant fit - $\chi^2$ = {chir2:.3f}', linewidth=linewidth)

radiiS, omegaK_yearS = kepler(Rs)
ax.plot(radiiS, omegaK_yearS, '.', label="Keplerian angular velocity")

#plot pseudoobservations
ax = axs[1]
R = Ro
v_obs = np.asarray(vmo).flatten()
v_err = np.asarray(v_stdo).flatten()

rs1, chi1, chir1, p1, A, devA = fit_vel(R, v_obs, v_err, vkep)
rs2, chi2, chir2, p2, B, devB = fit_vel(R, v_obs, v_err, vcost)

# line for the velocities
ax.plot(R, v_obs, '-', linewidth=linewidth,
        label=f"Measured speed")
# shaded error region
ax.fill_between(R, v_obs - v_err, v_obs + v_err, alpha=0.2, color=ax.lines[-1].get_color(), label="Error bar")  # use same color as line
ax.plot(R, vkep(R, A), '-.', label=fr"Keplerian fit - $\chi^2$ = {chir1:.3f}", linewidth=linewidth)
ax.plot(R, vcost(R, B), '--', label=fr"Constant fit - $\chi^2$ = {chir2:.3f}", linewidth=linewidth)

radiiO, omegaK_yearO = kepler(Ro)
ax.plot(radiiO, omegaK_yearO, '.', label="Keplerian angular velocity")


ax_lab = r"$\frac{d\phi}{dt} = \Omega(R)$ [radians/year]"
for i, ax in enumerate(axs):
    ax.set_xlabel('R [AU]', size=labelfontsize)
    ax.set_ylabel(ax_lab, size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)

    ax.text(
        0.5, 0.98, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=labelfontsize
    )
    ax.legend(fontsize=legfontsize)
    ax.grid(True)


plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()