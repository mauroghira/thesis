import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import sys
from astropy.io import fits
from scipy.ndimage import map_coordinates

folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

hdul = fits.open(name)
image_data = hdul[0].data
hdul.close()
image = image_data[0, 0, 0, :, :]  # select first frame


# 1. Shift image to center
ny, nx = image.shape
x0, y0 = nx/2.0, ny/2.0
X_full = np.arange(nx) - x0
Y_full = np.arange(ny) - y0
X_grid, Y_grid = np.meshgrid(X_full, Y_full)

i = np.radians(15)
deproj_factor = 1.0 / np.cos(i)
Yd_grid = Y_grid * deproj_factor
Xd_grid = X_grid

coords = np.array([Xd_grid, Yd_grid])
deproj_image = map_coordinates(image, coords, order=1, mode='reflect').reshape(ny, nx)

plt.figure(figsize=(8,8))   
plt.imshow(deproj_image, cmap='inferno', origin='lower')
plt.legend()
plt.title("Deprojected Spiral Image")
plt.show()