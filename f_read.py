#file with the functions to auto read the data

from astropy.io import fits #library to analyse fits files
import sys
import os
import numpy as np

#############
#===========================================================
############# function to read image data
def read(arg):
    if len(arg)==2 and "inc" in arg[1]:
        # If the input is a FITS file, read it
        outfile, image, label, pixel_size = read_fits(arg)

    #for the hydrodynamical simulations give the path massratio filename
    elif len(arg)==3:
        outfile, image, label, pixel_size = read_pix(arg)

    else:
        print("Invalid input. Please provide a valid FITS file or simulation data file.")
        sys.exit(1)

    return  outfile, image, label, pixel_size

#############
#===========================================================
############# function to read FITS or pix
def read_fits(arg):
    folder = "~/thesis/Spiral_pattern/"+arg[1]
    file="data_1300/RT.fits.gz"
    name = folder+file
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1])

    hdul = fits.open(name)
    image_data = hdul[0].data
    hdul.close()

    image = image_data[0, 0, 0, :, :]  # select first frame
    label = "Flux [W/(m⁻² pixel⁻¹)]"
    pixel_size = 320/image.shape[0] # AU
    #image = deproject_image(image, 0)

    return  outfile, image, label, pixel_size


def read_pix(arg):
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/sim_ana/")
    path = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/"+arg[2])
    image = np.loadtxt(path, dtype=float)
    label = "log column density [g/Cm⁻²]"
    pixel_size = 320/image.shape[0]  # AU

    return  outfile, image, label, pixel_size 


#############
#===========================================================
############# function to read single rphi file
def read_single(input_file):
    # Read array from input file (two columns: R and phi, use 'nan' for missing phi)
    data = np.loadtxt(input_file, dtype=float)
    R = data[:, 0]
    phi = data[:,1]

    # Indices of valid (non-nan) phi values
    valid = ~np.isnan(phi)
    R_valid = R[valid]
    phi_valid = phi[valid]
    data_valid = np.column_stack((R_valid, phi_valid))
    print("size: ",data_valid.size)

    return data_valid


#############
#===========================================================
############# function to read multiple rphi file
def read_mult(dire, arm, st=1):
    all_data = []
    for i in range(0, 11, st):
        input_file = dire + str(i) + "_" + arm + ".txt"
        if os.path.exists(input_file):
            data = read_single(input_file)
            all_data.append(data)
        else:
            print(f"File {input_file} not found, skipping.")

    print("len ",len(all_data))
    return all_data


#############
#===========================================================
############# function to read files formatted in col
def read_R_data_file(filename):
    """
    Reads a file with columns [x, y1, y2, ...].
    Returns:
        x : 1D numpy array
        ys : list of 1D numpy arrays, one for each y column
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    ys = [data[:, i] for i in range(1, data.shape[1])]
    return x, ys


#############
#===========================================================
############# function to deproject the image
from scipy.ndimage import map_coordinates
def deproject_image(image, inc_deg):
    """
    l'immagine è giusta se moltiplico per coseno.. non so perché hahah
    """
    inc_rad = np.deg2rad(inc_deg)
    # Create coordinate grid
    ny, nx = image.shape
    y, x = np.indices((ny, nx))
    x0, y0 = nx / 2, ny / 2
    x = x - x0
    y = y - y0

    # Deproject: stretch y axis by 1/cos(inc)
    y_deproj = y * np.cos(inc_rad)
    # Map back to pixel coordinates
    x_new = x + x0
    y_new = y_deproj + y0

    # Interpolate the image at new coordinates
    coords = np.array([y_new.flatten(), x_new.flatten()])
    deproj_img = map_coordinates(image, coords, order=1, mode='nearest').reshape(image.shape)

    return deproj_img