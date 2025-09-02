#file with the functions to auto read the data

from astropy.io import fits #library to analyse fits files
import sys
import os
import numpy as np

from f_gen import *

#############
#===========================================================
############# function to read image data
def read(arg):
    if len(arg)!=3:
        print("Invalid input. analyse.py ratio inc_X/yN or year")
        sys.exit(1)
    
    if "inc" in arg[2]:
        # If the input is a FITS file, read it
        outfile, image, label, pixel_size, vmin = read_fits(arg)

    #for the hydrodynamical simulations give the path massratio filename
    else:
        outfile, image, label, pixel_size, vmin = read_pix(arg)

    return  outfile, image, label, pixel_size, vmin

#############
#===========================================================
############# function to read FITS or pix
def read_fits(arg):
    folder = "~/thesis/Spiral_pattern/"+arg[1]+"/"+arg[2]+"/"
    file="data_1300/RT.fits.gz"
    name = folder+file
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/obs_data/")

    hdul = fits.open(name)
    image_data = hdul[0].data
    hdul.close()

    image = image_data[0, 0, 0, :, :]  # select first frame
    label = "Flux [W m⁻² pixel⁻¹]"
    pixel_size = 320/image.shape[0] # AU

    dif = mod_img(image, pixel_size)
    rot = np.rot90(dif, k=1)
    #image = deproject_image(image, 0)

    return  outfile, rot, label, pixel_size, -2e-21


def read_pix(arg):
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/sim_ana/")
    path = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/img_"+arg[1]+"_"+arg[2]+".pix")
    image = np.loadtxt(path, dtype=float)
    label = "log column density [g Cm⁻²]"
    pixel_size = 320/image.shape[0]  # AU

    return  outfile, image, label, pixel_size, None


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
############# function to filter out the constant pattern
def mod_img(image, px):
    #"""
    #compute the radial profile
    ny, nx = image.shape
    y, x = np.indices((ny, nx))   # full 2D coordinate grids
    r, phi = xy_to_rphi(y, x, image.shape[0])
    
    # Mask: keep only pixels inside the disk
    r_max_disk = (min(nx, ny) *np.sqrt(2) // 2)  # or another definition of disk radius
    mask = r <= r_max_disk

    #"""
    bin_centers, radial_mean = radial_average(image[mask], r[mask])
    # Interpolate values back to all r
    values = np.interp(r.ravel(), bin_centers, radial_mean, left=np.nan, right=np.nan)
    radial_img = values.reshape(image.shape)
    """
    bin_centers, radial_mean, r_bin_index = radial_average_masked(image, r, mask=mask)
    # Interpolate radial_mean to each valid pixel's radius (use bin centers)
    # Use np.interp on radii for masked pixels only.
    valid_r = r[mask]
    # For interpolation we need to skip bins that are nan. Create arrays of valid bins:
    valid_bins = ~np.isnan(radial_mean)
    if valid_bins.sum() < 2:
        raise RuntimeError("Not enough radial bins with data to interpolate.")
    interp_centers = bin_centers[valid_bins]
    interp_values = radial_mean[valid_bins]

    interp_values_for_pixels = np.interp(valid_r, interp_centers, interp_values,
                                         left=np.nan, right=np.nan)
    radial_img = np.full_like(image, np.nan, dtype=float)
    radial_img[mask] = interp_values_for_pixels
    #"""

    diff = image-radial_img

    return diff


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