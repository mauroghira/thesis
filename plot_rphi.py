import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_read import *

titlefontsize, labelfontsize, tickfontsize, legfontsize = 20, 16, 14, 12
markersize, linewidth = 6, 1

if len(sys.argv) != 3:
    print("Usage: python plot_rphi.py ratio arm")
    sys.exit(1)

dt = 10
ratio = sys.argv[1]
arm = sys.argv[2]
zero, rr = ratio.rsplit("0", 1)

outfile = "~/thesis/img_ts/"+ratio+"_phi_map"
base = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/")
fine = "_int_"+str(dt)+"_phi.txt"

#add data from simulations
if arm == "top":
    file = base+"sim_top"+fine
else:
    file = base+"sim_bot"+fine
Rs, phis = read_R_data_file(file)

#add data from observations
if arm == "top":
    file = base+"mc_top"+fine
else:
    file = base+"mc_bot"+fine
Ro, phio = read_R_data_file(file)


fig, axs = plt.subplots(1, 2, figsize=(17, 9))
title = 'q=0.'+str(rr)+fr', '+arm+' arm'

#èòot simulation datta
for i, p in enumerate(phis):
    ax = axs[0]
    R = Rs

    # line for the velocities
    ax.plot(R, p, '-', linewidth=linewidth, label=f"Year {i*dt}")

#plot observation data
for i, p in enumerate(phio):
    ax = axs[1]
    R = Ro

    # line for the velocities
    ax.plot(R, p, '-', linewidth=linewidth, label=f"Year {i*dt}")

ax_lab = r"$\varphi$ [radians]"
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
    # Set y ticks in multiples of π/2 using ax
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda val, pos: (
                f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
            )
        )
    )
    
    ax.legend(fontsize=legfontsize)
    ax.grid(True)

plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()