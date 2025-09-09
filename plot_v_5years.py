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
    print("Usage: python plot_v_5years.py ratio arm")
    sys.exit(1)

dt = 5
ratio = sys.argv[1]
arm = sys.argv[2]
zero, rr = ratio.rsplit("0", 1)

outfile = "~/thesis/img_ts/"+ratio+"_vel_5"
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
title = 'q=0.'+str(rr)+fr', '+arm+' arm'

for j, v in enumerate(vms):
    #plot simulation datta
    ax = axs[0]
    R = Rs
    v_sim = np.asarray(v).flatten()

    label = fr"Year {j*dt} $\rightarrow$ {j*dt+dt}"
    if j*dt ==10:
        label = fr"Year 0 $\rightarrow$ 10"
    # line for the velocities
    ax.plot(R, v_sim, '-', linewidth=linewidth, label=label)

radiiS, omegaK_yearS = kepler(Rs)
ax.plot(radiiS, omegaK_yearS, '.', label="Keplerian angular velocity")

#plot pseudoobservations
for j, v in enumerate(vmo):
    ax = axs[1]
    R = Ro
    v_obs = np.asarray(v).flatten()

    label = fr"Year {j*dt} $\rightarrow$ {j*dt+dt}"
    if j*dt ==10:
        label = fr"Year 0 $\rightarrow$ 10"
    # line for the velocities
    ax.plot(R, v_obs, '-', linewidth=linewidth, label=label)

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