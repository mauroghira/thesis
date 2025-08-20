from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_fits_file(file_path):
    """
    Reads a FITS file and returns the data and header information.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    tuple: A tuple containing the data and header of the FITS file.
    """
    with fits.open(file_path) as hdul:
        hdul.info()  # Print information about the HDU list
        data = hdul[0].data
        header = hdul[0].header
    return data, header

folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

# Open the FITS file
hdul = fits.open(name)

# FITS files often contain multiple Header/Data Units (HDUs)
# Usually the image data is in the primary HDU (index 0)
image_data = hdul[0].data

# Close the file after loading the data
hdul.close()

"""
#plot all the 8 images
print(image_data.shape)     # confirm shape
print(type(image_data))     # usually a numpy.ndarray
print(hdul[0].header)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
titles = [
    "I = total flux", "Q", "U", "V",
    "direct star light", "scattered star light",
    "direct thermal emission", "scattered thermal emission"
]

for i, ax in enumerate(axes.flat):
    img = image_data[i, 0, 0, :, :]
    im = ax.imshow(img, cmap="inferno", origin="lower")
    ax.set_title(titles[i])
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Flux [W m⁻² pixel⁻¹]")
plt.tight_layout()
plt.savefig("all_flux_components.pdf", bbox_inches="tight")
plt.show()
"""

path = os.path.expanduser(folder+"flux_7.jpg")

#plot and work only on the needed image
image_2d = image_data[0, 0, 0, :, :]  # select first frame
plt.imshow(image_2d, cmap="inferno", origin="lower")
plt.colorbar(label="Flux [W/(m⁻² pixel⁻¹)]")
plt.title("Total flux")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.savefig(path, bbox_inches="tight")
plt.show()