import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits #library to analyse fits files
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import sys
import os
from scipy.signal import find_peaks

dati = []
for i, arg in enumerate(sys.argv):
    if i>0:
        path = os.path.expanduser(arg)
        image = np.loadtxt(path, dtype=float)
        dati.append(image)

pixel_size = 320/image.shape[0]  # AU
hh = 3.5
allpeaks = []
styles = ['o', 's', '^', 'v', 'D', 'x', '*', 'p', 'h', '+', 'd']

plt.figure(figsize=(10, 10))
for j, image in enumerate(dati):
    peaks = []
    for i in range(image.shape[0]):
        peaks_in_one_trace, _ = find_peaks(image[i, :], height=hh)
        
        # Store peaks for each trace
        if peaks_in_one_trace.size > 0:
            for p in peaks_in_one_trace:
                peaks.append((i, p))
    allpeaks.append(peaks)

    # Select from image the peaks found
    rows, cols = zip(*peaks)  # unzip
    spiral = np.zeros_like(image)
    spiral[rows, cols] = image[rows, cols]
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    #rphi map
    rows = np.array(rows)
    cols = np.array(cols)
    ny, nx = image.shape
    x0, y0 = nx/2, ny/2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    #plot the R-phi map of the peaks
    plt.scatter(r, phi, s=1, alpha=0.5, marker=styles[j % len(styles)], label=f"Dataset {j+1}")

plt.title("R-$\phi$ Map of Peaks")
plt.xlabel("Radius (AU)")
plt.ylabel("$\phi$ (radians)")
plt.legend()
plt.grid()
#plt.savefig(outfile, bbox_inches="tight")
plt.show()