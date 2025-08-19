import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits #library to analyse fits files
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import sys
import os
from scipy.signal import find_peaks

# 3. Define logarithmic spiral model: r = a * exp(b * theta)
def log_spiral(theta, a, b):
    return a * np.exp(b * theta)

folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

hdul = fits.open(name)
image_data = hdul[0].data
hdul.close()

image = image_data[0, 0, 0, :, :]  # select first frame

#image = img[75:425, 75:425]  # crop image to reduce size

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

rows = np.array(rows)
cols = np.array(cols)

# 2. Convert to polar coordinates (center = image center)
ny, nx = image.shape
x0, y0 = nx/2, ny/2
x = cols - x0
y = rows - y0

r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Sort by radius
sort_idx = np.argsort(r)
r = r[sort_idx]
theta = theta[sort_idx]

# Unwrap theta so it grows continuously along the spiral
theta = np.unwrap(theta)

"""
# Bin theta
nbins = 500
bins = np.linspace(theta.min(), theta.max(), nbins)

# Compute median r in each bin (robust to outliers)
r_median, _, _ = binned_statistic(theta, r, statistic='median', bins=bins)

# Use bin centers as theta
theta_centers = 0.5 * (bins[1:] + bins[:-1])

# Remove bins with no points (NaN)
mask = ~np.isnan(r_median)
theta_rif = theta_centers[mask]
r_rif = r_median[mask]

p0 = [r_rif.min(), (np.log(r_rif.max()) - np.log(r_rif.min())) / (theta_rif.max() - theta_rif.min())]

popt, pcov = curve_fit(log_spiral, theta_rif, r_rif, p0=p0, maxfev=5000)

"""
# Initial guess for (a, b)
p0 = [r.min(), 0.1]

# Fit
popt, pcov = curve_fit(log_spiral, theta, r, p0=p0, maxfev=5000)
#"""

a_fit, b_fit = popt
print("Fitted spiral parameters: a =", a_fit, ", b =", b_fit)


# 4. Reconstruct fitted spiral
theta_rif = np.linspace(theta.min(), theta.max()*0.68, 10000)
r_fit = log_spiral(theta_rif, *popt)

# Convert back to Cartesian
x_fit = r_fit * np.cos(theta_rif) + x0
y_fit = r_fit * np.sin(theta_rif) + y0

# 5. Plot
plt.imshow(image, cmap="inferno", origin="lower")
plt.plot(cols, rows, 'r.', markersize=1, alpha=0.3, color="red", label="Detected pixels")
plt.plot(x_fit, y_fit, 'c-', linewidth=2, color="green", label="Fitted spiral")
plt.legend()
plt.show()