import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_read import *

titlefontsize, labelfontsize, tickfontsize, legfontsize =20, 16, 14, 13
markersize, linewidth = 6, 1

if len(sys.argv) != 3:
    print("Usage: python plot_v_fits.py ratio Arm")
    sys.exit(1)

dt = 10
ratio = sys.argv[1]
Arm = sys.argv[2]
arm = Arm.lower()
zero, rr = ratio.rsplit("0", 1)


outfile = "~/thesis/img_ts/"+ratio+"_vel_fits"
base = "~/thesis/Spiral_pattern/"+ratio+"/results/"
fine = "_"+arm+"_int_"+str(dt)+"_vel.txt"

Rs, vvs = [], []

file = os.path.expanduser(base+"sim"+fine)
R, vels = read_R_data_file(file)
Rs.append(R)
vvs.append(vels)

file = os.path.expanduser(base+"mc"+fine)
R, vels = read_R_data_file(file)
Rs.append(R)
vvs.append(vels)


fig, axs = plt.subplots(1, 2, figsize=(17, 9))
title = 'q=0.'+str(rr)+' - '+Arm+fr' arm - Year $0 \rightarrow 10$'
names=["Simulations", "Pseudo-observations"]
fig.suptitle(title, size=titlefontsize)

for i, v in enumerate(vvs):
    axs[i].plot(Rs[i], vvs[i][0], '-', lw=linewidth, label="Computed velocity")
    


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


plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()