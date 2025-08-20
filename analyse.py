from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import find_peaks 
from fuctions import *

#for the fits files simply give the path massratio/inclination/year/
if len(sys.argv)==2 and "inc" in sys.argv[1]:
    # If the input is a FITS file, read it
    folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
    file="data_1300/RT.fits.gz"
    name = folder+file
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"flux_7.jpg")

    hdul = fits.open(name)
    image_data = hdul[0].data
    hdul.close()

    image = image_data[0, 0, 0, :, :]  # select first frame
    label = "Flux [W/(m⁻² pixel⁻¹)]"
    pixel_size = 300/image.shape[0] # AU
    hh = 0.2e-19

#for the hydrodynamical simulations give the path massratio filename
elif len(sys.argv)==3:
    # If the input is a text file, read it
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/sim_ana/y0_peak.jpg")
    path = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/"+sys.argv[2])
    image = np.loadtxt(path, dtype=float)
    label = "log column density [g/Cm⁻²]"
    pixel_size = 320/image.shape[0]  # AU
    hh = 3.5

else:
    print("Invalid input. Please provide a valid FITS file or simulation data file.")
    sys.exit(1)


peaks = []
# Find peaks in the image data
for i in range(image.shape[0]):
    peaks_in_one_trace, _ = find_peaks(image[i, :], height=hh)
    
    # Store peaks for each trace
    if peaks_in_one_trace.size > 0:
        for p in peaks_in_one_trace:
            peaks.append((i, p))

# Select from image the peaks found
rows, cols = zip(*peaks)  # unzip
spiral = np.zeros_like(image)
spiral[rows, cols] = image[rows, cols]

extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

"""
plt.imshow(spiral, cmap="inferno", origin="lower", extent=extent)
plt.colorbar(label=label)
plt.title("Peaks only")
plt.xlabel("x (AU)")
plt.ylabel("Y (AU)")
#plt.savefig(path, bbox_inches="tight")
plt.show()
#"""

rows = np.array(rows)
cols = np.array(cols)
r, phi = xytorphi(rows, cols, image.shape[0])
rphi_peaks = np.column_stack((r, phi))

"""
#plot the R-phi map of the peaks
plt.figure(figsize=(8, 6))
plt.scatter(r, phi, s=1, c='blue', alpha=0.5)
plt.title("R-$\phi$ Map of Peaks")
plt.xlabel("Radius (AU)")
plt.ylabel("$\phi$ (radians)")
plt.grid()
#plt.savefig(outfile, bbox_inches="tight")
plt.show()
#"""

#detect two particular spiral arms
#peaks = np.array(peaks)
rphi_neighbors = vicini(rphi_peaks, image.shape[0], bound=3.4, neig=10, iter=10)

xn, yn = rphitoxy(rphi_neighbors, image.shape[0])
xy_neighbors = np.column_stack((xn, yn))

scaled_neighbors = (xy_neighbors-image.shape[0]/2) * pixel_size
#scaled_neighbors = (neighbors[0]*pixel_size)

fig = plt.figure(figsize=(10, 10))
plt.scatter(xy_neighbors[:,0], xy_neighbors[:,1], color="lime",  s=60, edgecolor="k", label="Neighbors")
#plt.scatter(scaled_neighbors[:,0], scaled_neighbors[:,1], color="lime",  s=10, edgecolor="k", label="Neighbors")
#plt.imshow(image, cmap="inferno", origin="lower", extent=extent)
#plt.imshow(spiral, cmap="inferno", origin="lower", extent=extent)
#plt.colorbar(label="Log column density [g/cm^2]")
plt.legend()
#plt.savefig("prima_spirale.jpg", bbox_inches="tight")
plt.show()