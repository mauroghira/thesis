from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import find_peaks 


folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

hdul = fits.open(name)
image_data = hdul[0].data
hdul.close()

image = image_data[0, 0, 0, :, :]  # select first frame

peaks = []
# Find peaks in the image data
for i in range(image.shape[0]):
    peaks_in_one_trace, _ = find_peaks(image[i, :], height=0.2e-19)
    
    # Store peaks for each trace
    if peaks_in_one_trace.size > 0:
        for p in peaks_in_one_trace:
            peaks.append((i, p))

# Select from image the peaks found
rows, cols = zip(*peaks)  # unzip
spiral = np.zeros_like(image)
spiral[rows, cols] = image[rows, cols]

plt.imshow(spiral, cmap="inferno", origin="lower")
plt.colorbar(label="Flux [W/(m⁻² pixel⁻¹)]")
plt.title("Peaks only")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
#plt.savefig(path, bbox_inches="tight")
plt.show()