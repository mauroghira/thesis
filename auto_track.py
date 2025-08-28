import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import find_peaks 
from fuctions import *

if len(sys.argv) != 2:
    print("Usage: python auto_track.py mass_ratio")
    sys.exit(1)

ratio = sys.argv[1]

images = []
for i in range(20,31):
    input_file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio +"/m03_000"+str(i)+"_logcolumndensitygcm2_proj.pix")
    if os.path.exists(input_file):
        image = np.loadtxt(input_file, dtype=float)
        images.append(image)
    else:
        print(f"File {input_file} not found, skipping.")

outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/sim_ana/")
label = "log column density [g/Cm⁻²]"
pixel_size = 320/544  # AU

val = float(input("threshold> "))

rm = float(input("R min> "))
rM = float(input("R max> "))
phim = float(input("phi min> "))
phiM = float(input("phi max> "))
name = input("top/bot> ")
lim = float(input("lim> "))

for i, image in enumerate(images):
    print("year ", i)

    peaks = find_2d_peaks(image, val)
    rows = np.array([peak[0] for peak in peaks])
    cols = np.array([peak[1] for peak in peaks])
    spiral = np.zeros_like(image)
    spiral[rows, cols] = image[rows, cols]

    rep = "y"
    while rep== "y":
        xy_neighbors = filter_peaks_by_rphi(peaks, image.shape[0], pixel_size, rm, rM, phim, phiM, lim)
        plot_neighbors(xy_neighbors, image, pixel_size, label)
        plt.show()

        rep = input("try again?> ")
        if rep == "y":
            lim = float(input("lim> "))

    file = outfile+str(i)+"_"+name+".txt"
    save_rphi(xy_neighbors, file, phiM, pixel_size, image.shape[0])