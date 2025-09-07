import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from f_read import *
from f_track import *

titlefontsize, labelfontsize, tickfontsize, legfontsize =20, 16, 14, 13
markersize, linewidth = 6, 1

if len(sys.argv) != 3:
    print("Usage: python plot_peaks.py ratio year")
    sys.exit(1)

ratio = sys.argv[1]
out = "~/thesis/img_ts/"

zero, rr = ratio.rsplit("0", 1)
y = sys.argv[2]
base = "~/thesis/Spiral_pattern/"+ratio

labels = ["Log column density [g/Cm²]",  "Flux [W/(m² pixel)]"]
title = f"q=0.{rr}, year {y}"

input_file = os.path.expanduser(base+"/img_"+ratio+"_"+y+".pix")
sim = np.loadtxt(input_file, dtype=float)

input_file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/inc_0/y"+y+"/data_1300/RT.fits.gz")
hdul = fits.open(input_file)
image_data = hdul[0].data
hdul.close()

obs = image_data[0, 0, 0, :, :]  # select first frame
obs = mod_img(obs) #to subtract radial map
rot = np.rot90(obs, k=1)

imgs = [sim, rot]
pixel_size = [320/544, 320/512]

nb = []
#find peaks
for i, image in enumerate(imgs):
    val = float(input("threshold> "))
    mx = float(input("max> "))
    peaks = find_2d_peaks(image, val, mx)
    xy_neighbors = filter_peaks_by_rphi(peaks, image.shape[0], pixel_size[i], 0, 100, -180, 180)
    scaled_neighbors = (xy_neighbors-(image.shape[0]-1)/2) * pixel_size[i]
    nb.append(scaled_neighbors)

exts = [[-(image.shape[1]-1) * 160/image.shape[1], (image.shape[1]-1) * 160/image.shape[1], -(image.shape[0]-1) * 160/image.shape[0], (image.shape[0]-1) * 160/image.shape[0]] for image in imgs]
vmins = [None, None]
vmins = [None, -1e-21]

fig, axs = plt.subplots(1, 2, figsize=(19, 9), sharey=True)
for i, ax in enumerate(axs):
    im = ax.imshow(imgs[i], cmap="inferno", vmin=vmins[i], origin="lower", extent=exts[i])
    
    # inset axes on top of each subplot
    cax = inset_axes(ax,
                     width="100%",   # full width of the subplot
                     height="5%",    # thickness of colorbar
                     loc="upper center",
                     bbox_to_anchor=(0, 0.05, 1, 1),  # shift slightly above
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(labels[i], fontsize=labelfontsize, labelpad=15)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    ax.set_xlabel("x [AU]", size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)

    ax.scatter(nb[i][:, 1], nb[i][:, 0], color="lime", s=1, label="Peaks")

    # panel title inside the image
    ax.text(
        0.5, 0.98, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=labelfontsize,
        color="white",
        bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2)
    )

    ax.legend(fontsize=legfontsize)

axs[0].set_ylabel("Y [AU]", size=labelfontsize)

axs[1].spines["left"].set_visible(False)
pos0 = axs[0].get_position()
pos1 = axs[1].get_position()
# glue them
axs[1].set_position([pos0.x1, pos1.y0, pos1.width, pos1.height])

for label in axs[1].get_xticklabels():
    label.set_horizontalalignment("left")
for label in axs[0].get_xticklabels():
    label.set_horizontalalignment("right")

outfile = out + ratio +"_"+y+"_peaks"

plt.savefig(os.path.expanduser(outfile+".pdf"))
plt.savefig(os.path.expanduser(outfile+".png"), bbox_inches="tight")
plt.show()