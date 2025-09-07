import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_read import *

titlefontsize, labelfontsize, tickfontsize, legfontsize =20, 16, 14, 13
markersize, linewidth = 6, 1

if len(sys.argv) != 1:
    print("Usage: python plot_v_ratios.py")
    sys.exit(1)

dt = 10
#there should be only one velocity per ataset this way

ratios = [15, 25, 3, 1, 3]

out = "~/thesis/img_ts/"
base = "~/thesis/Spiral_pattern/"
fine = "_int_"+str(dt)+"_vel.txt"

Rs, vvs = [], []
#add data from simulations
for r in [15, 25, 3]:
    file = os.path.expanduser(base+"0"+str(r)+"/results/"+"sim_top"+fine)
    R, vels = read_R_data_file(file)
    Rs.append(R)
    vvs.append(vels)
for r in [1, 3]:
    file = os.path.expanduser(base+"0"+str(r)+"/results/"+"sim_bot"+fine)
    R, vels = read_R_data_file(file)
    Rs.append(R)
    vvs.append(vels)

Ro, vvo = [], []
#add data from observations
for r in [15, 25, 3]:
    file = os.path.expanduser(base+"0"+str(r)+"/results/"+"mc_top"+fine)
    R, vels = read_R_data_file(file)
    Ro.append(R)
    vvo.append(vels)
for r in [1, 3]:
    file = os.path.expanduser(base+"0"+str(r)+"/results/"+"mc_bot"+fine)
    R, vels = read_R_data_file(file)
    Ro.append(R)
    vvo.append(vels)


fig, axs = plt.subplots(1, 2, figsize=(17, 9))
title = fr'Year $0 \rightarrow 10$'
names=["Simulations", "Pseudo-observations"]

for i, v in enumerate(vvs):
    if i < 3:
        lbl = "q=0."+str(ratios[i])+" - Top arm"
    else:
        lbl = "q=0."+str(ratios[i])+" - Bottom arm"
    axs[0].plot(Rs[i], vvs[i][0], '-', lw=linewidth, label=lbl)
    axs[1].plot(Ro[i], vvo[i][0], '-', lw=linewidth, label=lbl)

radiiS, omegaK_yearS = kepler(Rs)
axs[0].plot(radiiS, omegaK_yearS, '.', label="Keplerian angular velocity")
radiiO, omegaK_yearO = kepler(Ro)
axs[1].plot(radiiO, omegaK_yearO, '.', label="Keplerian angular velocity")

ax_lab = r"$\frac{d\phi}{dt} = \Omega(R)$ [radians/year]"
for i, ax in enumerate(axs):
    ax.set_xlabel('R [AU]', size=labelfontsize)
    ax.set_ylabel(ax_lab, size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)
    ax.set_title(names[i], size=titlefontsize)
    ax.legend()
    ax.grid(True)

    ax.text(
        0.5, 0.98, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=labelfontsize,
        color="white",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2)
    )


outfile = out +"vel_different_ratios"
plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()