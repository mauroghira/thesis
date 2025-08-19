from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#filename = "~/thesis/Spiral_pattern/"+sys.argv[1]
path = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1])

width, height = 512, 512  # you must know this
dtype = np.uint8          # or np.float32, etc.

data = np.fromfile(path, dtype=dtype)
image = data.reshape((height, width))

#plot and work only on the needed image
plt.imshow(image, cmap="inferno", origin="lower")
plt.colorbar(label="Flux [W/(m⁻² pixel⁻¹)]")
plt.title("Total flux")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
#plt.savefig(path, bbox_inches="tight")
plt.show()