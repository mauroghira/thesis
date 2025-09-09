import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_read import *

titlefontsize, labelfontsize, tickfontsize, legfontsize =20, 16, 14, 13
markersize, linewidth = 6, 1

#plot spirals at different times with tracks

if len(sys.argv) != 3:
    print("Usage: python plot_tracks.py ratio dt")
    sys.exit(1)

ratio = sys.argv[1]
dt = int(sys.argv[2])

zero, rr = ratio.rsplit("0", 1)
out = "~/thesis/img_ts/"
base = "~/thesis/Spiral_pattern/"+ratio

labels = ["Log column density [g/Cm²]",  "Flux [W/(m² pixel)]"]

imgs = []
for i in range(0,11,dt):
    input_file = os.path.expanduser(base+"/img_"+ratio+"_"+str(i)+".pix")
    sim = np.loadtxt(input_file, dtype=float)

    input_file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/inc_0/y"+str(i)+"/data_1300/RT.fits.gz")
    hdul = fits.open(input_file)
    image_data = hdul[0].data
    hdul.close()

    obs = image_data[0, 0, 0, :, :]  # select first frame
    obs = mod_img(obs) #to subtract radial map
    rot = np.rot90(obs, k=1)

    im = [sim, rot]
    imgs.append(im)

# each row in imgs has sim vs mc at fix time
exts = [[-(image.shape[1]-1) * 160/image.shape[1], (image.shape[1]-1) * 160/image.shape[1], -(image.shape[0]-1) * 160/image.shape[0], (image.shape[0]-1) * 160/image.shape[0]] for image in imgs[0]]
vmins = [None, -1e-21]

#thi will be an extraindex to distinguish bottom and top arm
colors = ["yellow", "blue"]

# here the index wil then run on the columns of imgs
incipit = ["sim", "mc"]
Rs = [] # radius are common for all phis in sim and all phis in mc but are different between the two
phiss = []
for i in incipit:
    file = os.path.expanduser(base+"/results/"+i+"_top_int_"+str(dt)+"_phi.txt")
    R, phis = read_R_data_file(file)
    Rs.append(R)
    phiss.append(phis)

Rs2 = [] # radius are common for all phis in sim and all phis in mc but are different between the two
phiss2 = []
for i in incipit:
    file = os.path.expanduser(base+"/results/"+i+"_bot_int_"+str(dt)+"_phi.txt")
    R, phis = read_R_data_file(file)
    Rs2.append(R)
    phiss2.append(phis)


nrows = len(imgs)
ncols = 2

fig = plt.figure(figsize=(18,18))

# margins
left, right, bottom, top = 0.08, 0.98, 0.03, 0.97
nrows, ncols = len(imgs), 2

# --- grid sizing ---
cbar_h = 0.02
cbar_label_pad = 0.02  # extra space for label
usable_top = top - cbar_h - cbar_label_pad
col_w = (right - left) / ncols
usable_height = usable_top - bottom
ax_h = usable_height / nrows
ax_w = col_w

axs = np.empty((nrows, ncols), dtype=object)

for i in range(nrows):
    for j in range(ncols):
        # compute left and bottom (i=0 is top row)
        left_x = left + j * col_w
        bottom_y = bottom + (nrows - 1 - i) * ax_h

        ax = fig.add_axes([left_x, bottom_y, ax_w, ax_h])
        axs[i, j] = ax

        # image (imgs[i][j] must exist)
        im = ax.imshow(imgs[i][j], cmap="inferno", vmin=vmins[j],
                       origin="lower", extent=exts[j])

        # overlay spiral arm for this column/row if available
        try:
            phi = phiss[j][i]     # phiss organized per column
            x, y = rphi_to_xy(Rs[j], phi, 0)
            ax.plot(x, y, "-", color=colors[0], lw=linewidth, label="Top arm")
        except Exception:
            pass
        # overlay spiral arm for this column/row if available
        try:
            phi = phiss2[j][i]     # phiss organized per column
            x, y = rphi_to_xy(Rs2[j], phi, 0)
            ax.plot(x, y, "-", color=colors[1], lw=linewidth, label="Bottom arm")
        except Exception:
            pass

        # panel title inside image
        year = i * dt
        ax.text(0.5, 0.98, f"q=0.{rr}, year {year}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=labelfontsize, color="white",
                bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2))

        # ticks/labels: only bottom row has x-labels, only left column has y-labels
        if i != nrows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x [AU]", size=labelfontsize)
            ax.tick_params(axis="x", labelsize=tickfontsize)

        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("y [AU]", size=labelfontsize)
            ax.tick_params(axis="y", labelsize=tickfontsize)

        legend = ax.legend(fontsize=legfontsize, facecolor="black", edgecolor="none", framealpha=0.3)
        ax.set_aspect("equal", "box")
        for text in legend.get_texts():
            text.set_color("white")   # set legend text color


# -------------------------
# single colorbar per column, placed above the first-row axes
# -------------------------
for j in range(ncols):
    pos = axs[0, j].get_position()                 # position of top-row axis in figure coords
    cax = fig.add_axes([pos.x0, pos.y1, pos.width, cbar_h])
    im_top = axs[0, j].images[0]                   # the imshow from top row
    cbar = fig.colorbar(im_top, cax=cax, orientation="horizontal")
    cbar.set_label(labels[j], fontsize=labelfontsize, labelpad=10)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")


# nudge bottom-row xtick alignments so seam labels don't collide
for label in axs[-1, 0].get_xticklabels():
    label.set_horizontalalignment("right")
for label in axs[-1, 1].get_xticklabels():
    label.set_horizontalalignment("left")


outfile = out + ratio +"_tracks"
plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()