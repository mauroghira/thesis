import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits #library to analyse fits files
import sys
import os
from scipy.signal import find_peaks
from scipy.spatial import KDTree
import csv
from scipy.ndimage import map_coordinates
from scipy.ndimage import maximum_filter

#############
#===========================================================
############# function to read data

def read(arg):
    if len(arg)==2 and "inc" in arg[1]:
        # If the input is a FITS file, read it
        folder = "~/thesis/Spiral_pattern/"+arg[1]
        file="data_1300/RT.fits.gz"
        name = folder+file
        outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1])

        hdul = fits.open(name)
        image_data = hdul[0].data
        hdul.close()

        image = image_data[0, 0, 0, :, :]  # select first frame
        label = "Flux [W/(m⁻² pixel⁻¹)]"
        pixel_size = 300/image.shape[0] # AU
        image = deproject_image(image, 0)

    #for the hydrodynamical simulations give the path massratio filename
    elif len(arg)==3:
        # If the input is a text file, read it
        outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/sim_ana/")
        path = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/"+arg[2])
        image = np.loadtxt(path, dtype=float)
        label = "log column density [g/Cm⁻²]"
        pixel_size = 320/image.shape[0]  # AU

    else:
        print("Invalid input. Please provide a valid FITS file or simulation data file.")
        sys.exit(1)

    return  outfile, image, label, pixel_size


#############
#===========================================================
############# function to deproject the image
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


#############
#===========================================================
############# function to find the spiral arms in the image
def find_2d_peaks(image, threshold=0):
    # Apply a maximum filter to find local maxima
    neighborhood = np.ones((3, 3))
    local_max = (image == maximum_filter(image, footprint=neighborhood))
    detected_peaks = np.argwhere(local_max & (image > threshold))
    return detected_peaks


#############
#===========================================================
############# function to select an arm
def filter_peaks_by_rphi(peaks, image_size, px_size, r_min=0, r_max=100, phi_min=-90, phi_max=90, lim=3):
    """
    Filter peaks by polar coordinate conditions.
    peaks: array of (row, col) pixel coordinates
    image_size: size of the image (assumed square)
    r_min, r_max: min/max radius (in AU)
    phi_min, phi_max: min/max angle (in deg)
    Returns: filtered array of peaks (pixel coordinates)
    """

    phi_max = np.deg2rad(phi_max)
    phi_min = np.deg2rad(phi_min)
    r_min = r_min / px_size
    r_max = r_max / px_size

    rows, cols = peaks[:, 0], peaks[:, 1]
    r, phi = xy_to_rphi(rows, cols, image_size)
    
    # to properly track the spiral arm from the top
    if phi_max > np.pi:
        phi = (phi + 2*np.pi) % (2*np.pi)

    mask = np.ones_like(r, dtype=bool)
    if r_min < r_max:
        mask &= (r >= r_min)
        mask &= (r <= r_max)
    if phi_min < phi_max:
        mask &= (phi >= phi_min)
        mask &= (phi <= phi_max)
    partial = peaks[mask]

    # Sort peaks by increasing phi (counterclockwise order)
    r, phi = xy_to_rphi(partial[:, 0], partial[:, 1], image_size)
    sort_idx = np.argsort(phi)
    sorted_peaks = partial[sort_idx]

    # Filter out points with radius too far from the average of their neighbors
    filtered_peaks = []
    r_sorted = r[sort_idx]
    phi_sorted = phi[sort_idx]

    angular_window = np.deg2rad(10)  # set window size to 10deg
    for i in range(len(sorted_peaks)):
        # Get indices of neighbors where the angle difference is within the angular window
        indices = [
            j for j in range(len(sorted_peaks))
            if j != i and abs(angle_diff(phi_sorted[j], phi_sorted[i])) <= angular_window
        ]
        
        if not indices:
            continue

        min_v = np.min([r_sorted[j] for j in indices])
        max_v = np.max([r_sorted[j] for j in indices])
        avg = (max_v+min_v)/2
        dif = max_v - min_v

        if r_sorted[i] >= avg or dif <= (lim/px_size):
            filtered_peaks.append(sorted_peaks[i])
        else:
            r_sorted[i] = r_sorted[i-1]
            x, y = rphi_to_xy(r_sorted[i-1], phi_sorted[i], image_size)
            sorted_peaks[i] = np.array(y,x)
    #"""
    
    return np.array(filtered_peaks)


#############
#===========================================================
############# function to measure angular difference

def angle_diff(phi_new, phi_old):
    """Return signed CCW difference in range (-π, π]."""
    dphi = phi_new - phi_old
    # wrap into [-π, π]
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    return dphi


#############
#===========================================================
############# function to convert coords

def xy_to_rphi(rows, cols, size):
    # Convert pixel coordinates to R-phi coordinates
    ny, nx = size-1, size-1
    x0, y0 = nx / 2, ny / 2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi

def rphi_to_xy(r, phi, size):
    x = r * np.cos(phi) + (size-1) / 2
    y = r * np.sin(phi) + (size-1) / 2

    return x, y


#############
#===========================================================
############# function to plot the original image or the peaks

def plot_image(image, pixel_size, label, path=""):
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(label=label)
    plt.title("Spiral Pattern")
    plt.xlabel("x (AU)")
    plt.ylabel("Y (AU)")
    #plt.show()


#############
#===========================================================
############# function to plot the spiral arm neighbors

def plot_neighbors(xy_neighbors, image, pixel_size, label, path=""):
    scaled_neighbors = (xy_neighbors-(image.shape[0]-1)/2) * pixel_size
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(scaled_neighbors[:, 1], scaled_neighbors[:, 0], color="lime", s=10, edgecolor="k", label="Single spial arm")
    #plt.plot(scaled_neighbors[:, 1], scaled_neighbors[:, 0], '-', color="lime", label="Single spial arm")
    plt.imshow(image, cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(label=label)
    plt.title("Spiral Pattern")
    plt.xlabel("x (AU)")
    plt.ylabel("Y (AU)")
    plt.legend()
    #plt.show()


#############
#===========================================================
############# function to plot the Rphi map of the peaks

def plot_rphi_map(r, phi, peaks_size):
    r = r * peaks_size  # scale radius
    plt.figure(figsize=(10, 10))
    plt.scatter(r, phi, c='black', s=5, label="Spiral Arms")
    plt.title("R-$\phi$ Map of Peaks")
    plt.xlabel("Radius (AU)")
    plt.ylabel("$\phi$ (radians)")

    # set y ticks in multiples of π/2
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid()
    #plt.show()


#############
#===========================================================
############# function to save the spiral arm

def save_rphi(points, file, phimax, scale, size):
    #scaling radius
    r, phi = xy_to_rphi(points[:,0], points[:,1], size)
    r = r * scale

    #if starting from the top, make the angle monotonous
    if phimax > 180:
        phi = (phi + 2*np.pi) % (2*np.pi)

    data = np.column_stack((r,phi))
    with open(file, 'w') as f:
        csv.writer(f, delimiter=' ').writerows(data)