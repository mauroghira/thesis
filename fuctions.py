import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits #library to analyse fits files
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import sys
import os
from scipy.signal import find_peaks
from scipy.spatial import KDTree


def read(arg):
    if len(arg)==2 and "inc" in arg[1]:
        # If the input is a FITS file, read it
        folder = "~/thesis/Spiral_pattern/"+arg[1]
        file="data_1300/RT.fits.gz"
        name = folder+file
        outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"flux_7.jpg")

        hdul = fits.open(name)
        image_data = hdul[0].data
        hdul.close()

        image = image_data[0, 0, 0, :, :]  # select first frame
        label = "Flux [W/(m⁻² pixel⁻¹)]"
        pixel_size = 300/image.shape[0] # AU
        hh = 0.2e-19

    #for the hydrodynamical simulations give the path massratio filename
    elif len(arg)==3:
        # If the input is a text file, read it
        outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/sim_ana/y0_peak.jpg")
        path = os.path.expanduser("~/thesis/Spiral_pattern/"+arg[1]+"/"+arg[2])
        image = np.loadtxt(path, dtype=float)
        label = "log column density [g/Cm⁻²]"
        pixel_size = 320/image.shape[0]  # AU
        hh = 3.5

    else:
        print("Invalid input. Please provide a valid FITS file or simulation data file.")
        sys.exit(1)

    return  outfile, image, label, pixel_size, hh

def vicini(points, size, bound=np.inf, neig=10, iter=10):
    #work on points in the R phi representation to avoid infinite loops
    tree = KDTree(points)
    max = np.argmax(points[:, 0])
    mean = np.mean(points[:, 0])
    query_point = points[max]  # pick one point
    #qp = np.array([])
    print(points[max], mean)

    all_ind = []
    # track the spiral by looping the neighbour search
    while query_point[0] > mean:
        #qp = np.append(qp, query_point)
        distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)  # k nearest neighbors
        
        mask = distances <= (distances.mean() + distances.std()*0.7)
        distances = distances[mask]
        indices   = indices[mask]
        ind_dmax = np.argmax(distances)
        
        #i assume the spiral to be monotonous in the R direction
        #so i can folow the arm by decreasing the radius
        Rmax, phimax = points[indices[ind_dmax]]
        while Rmax >= query_point[0]:
            distances = np.delete(distances, ind_dmax)
            indices = np.delete(indices, ind_dmax)
            if indices.size == 0:
                break
            ind_dmax = np.argmax(distances)
            Rmax, phimax = points[indices[ind_dmax]]
        if indices.size == 0:
            break
        query_point = points[indices[ind_dmax]]  # pick the farthest point
        #print("Query point:", query_point, "Distance:", distances[ind_dmax])

        all_ind.extend(indices.tolist())  # add all elements, not the array itself

    #remove duplicates
    all_ind = np.unique(all_ind)
    vic = points[all_ind]
    return vic

def xytorphi(rows, cols, size):
    # Convert pixel coordinates to R-phi coordinates
    ny, nx = size, size
    x0, y0 = nx / 2, ny / 2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi

def rphitoxy(scaled_neighbors, size):
    # conversion from polar to cartesian coordinates
    r, phi = scaled_neighbors[:, 0], scaled_neighbors[:, 1]
    x = r * np.cos(phi) + size / 2
    y = r * np.sin(phi) + size / 2

    return x, y