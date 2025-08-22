import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits #library to analyse fits files
import sys
import os
from scipy.signal import find_peaks
from scipy.spatial import KDTree
import csv

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
############# function to find the spiral arms in the image

def spiral_finder(image, hh):
    peaks = []
    # Find peaks in the image data
    for i in range(image.shape[0]):
        peaks_in_one_trace, _ = find_peaks(image[i, :], height=hh)
        
        # Store peaks for each trace
        if peaks_in_one_trace.size > 0:
            for p in peaks_in_one_trace:
                peaks.append((i, p))

    return np.array(peaks)


#############
#===========================================================
############# function to isolate one spiral arm

def neig(cart_points, size, bound=1, max_dr=1, start="b",
           bound_step=0.1, max_expansions=50):
    verbose=False
 
    r, phi = xy_to_rphi(cart_points[:, 0], cart_points[:,1], size)
    points = np.column_stack((r, phi))

    tree = KDTree(cart_points)

    # choose start half and map back to global indices
    if start == "b":
        partial_ix = np.nonzero(points[:, 1] < 0)[0]
    elif start == "t":
        partial_ix = np.nonzero(points[:, 1] > 0)[0]
    else:
        print("Invalid start point. Use 'b' for bottom or 't' for top.")
        return None

    if partial_ix.size == 0:
        print("No points in the selected half.")
        return None

    # pick the point with largest radius in that half
    max_idx_local = np.argmax(points[partial_ix, 0])
    current_idx = partial_ix[max_idx_local]
    mean_r = np.mean(points[:, 0])

    listed = set([current_idx])  # mark start as visited to avoid revisiting

    while points[current_idx][0] > mean_r:
        # choose a sane starting radius if user passed np.inf
        temp_bound = bound if np.isfinite(bound) else bound_step

        selected_indices = None
        while True:
            # 1) neighbors within current radius
            ball = tree.query_ball_point(cart_points[current_idx], r=temp_bound)

            # 2) drop already visited & self
            cand = [i for i in ball if i not in listed and i != current_idx]

            # 3) enforce maximum radial step
            if np.isfinite(max_dr):
                r0 = points[current_idx][0]
                cand = [i for i in cand if abs(points[i, 0] - r0) <= max_dr]

            # 4) check for counterclockwise wrapping
            phi0 = points[current_idx, 1]
            cand = [i for i in cand if angle_diff(points[i,1], phi0) > 0]

            if verbose:
                print(f"[r={points[current_idx][0]:.3f}, phi={points[current_idx][1]:.3f}] "
                      f"ball={len(ball)} after-visited={len(ball)-len([i for i in ball if i not in cand or i==current_idx])} "
                      f"after-radial={len(cand)} (temp_bound={temp_bound:.3f})")

            if cand:  # we have valid candidates after radial filtering
                selected_indices = cand
                break

            # grow the radius; note this won’t “defeat” max_dr since that’s independent of temp_bound
            temp_bound += bound_step
            if temp_bound > max_expansions:
                print("No neighbors satisfy conditions after many expansions — stopping.")
                print(f"Total collected: {len(listed)}")
                return points[list(sorted(listed))]

        # add *all* found neighbors to the visited set (your requirement)
        listed.update(selected_indices)

        # step to the farthest candidate (in [r,phi] Euclidean space)
        dists = np.linalg.norm(cart_points[selected_indices] - cart_points[current_idx], axis=1)
        next_idx = selected_indices[int(np.argmax(dists))]

        if verbose:
            print(f"→ step to idx {next_idx}: point={points[next_idx]} dist={dists.max():.4f}\n")

        # advance
        current_idx = next_idx

    print(f"Total collected: {len(listed)}")

    return points[list(sorted(listed))]


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
    ny, nx = size, size
    x0, y0 = nx / 2, ny / 2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi

def rphi_to_xy(scaled_neighbors, size):
    # conversion from polar to cartesian coordinates
    r, phi = scaled_neighbors[:, 0], scaled_neighbors[:, 1]
    x = r * np.cos(phi) + size / 2
    y = r * np.sin(phi) + size / 2

    return x, y


#############
#===========================================================
############# function to plot the original image or the peaks

def plot_image(image, pixel_size, label, path=""):
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]
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
    scaled_neighbors = (xy_neighbors-image.shape[0]/2) * pixel_size
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(scaled_neighbors[:, 0], scaled_neighbors[:, 1], color="lime", s=10, edgecolor="k", label="Single spial arm")
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

def save_rphi(points, file, start, scale):
    #scaling radius
    points[:, 0] = points[:, 0] * scale

    test = points.copy()
    #if starting from the top, make the angle monotonous
    if start=="t":
        """
        # method 1: loop
        test1 = points.copy()
        for i in range(test1.shape[0]):
            if test1[i, 1] < 0:
                test1[i, 1] += 2*np.pi
        """
        # method 2: vectorized
        test[:, 1] = (test[:, 1] + 2*np.pi) % (2*np.pi)

    # compare
    #plot_rphi_map(test[:,0], test[:,1], False)
    #plt.show()

    with open(file, 'w') as f:
        csv.writer(f, delimiter=' ').writerows(test)