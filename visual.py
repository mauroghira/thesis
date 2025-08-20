from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    plt.imshow(image, cmap="inferno", extent=extent, origin="lower")
    plt.colorbar(label="Flux [W/(m⁻² pixel⁻¹)]")
    plt.title("Total flux")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

#for the hydrodynamical simulations give the path massratio filename
elif len(sys.argv)==3:
    # If the input is a text file, read it
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/sim_ana/y5.jpg")
    path = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/"+sys.argv[2])
    image = np.loadtxt(path, dtype=float)
    label = "log column density [g/Cm⁻²]"
    pixel_size = 320/image.shape[0]  # AU
    hh = 3.5
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Disc density")

    images = [image, 10**image]

    for i, axs in enumerate(ax):
        pcm = axs.imshow(images[i], cmap="inferno", extent=extent, origin="lower")
        fig.colorbar(pcm, ax=axs, label="(log) column density [g/Cm⁻²]")
        axs.set_xlabel("x (AU)")
        axs.set_ylabel("Y (AU)")

    plt.tight_layout()
    #plt.savefig(outfile, bbox_inches="tight")
    plt.show()

else:
    print("Invalid input. Please provide a valid FITS file or simulation data file.")
    sys.exit(1)