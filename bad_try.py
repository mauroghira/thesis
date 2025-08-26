from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.signal import find_peaks

# Example dataset: random scatter + a "spiral arm"
np.random.seed(0)
random_points = np.random.rand(100, 2) * 10   # background noise
theta = np.linspace(0, 4*np.pi, 50)
spiral_x = theta * np.cos(theta)
spiral_y = theta * np.sin(theta)
spiral_points = np.column_stack([spiral_x + 5, spiral_y + 5])  # shift to center

# Combine into one dataset
points = np.vstack([random_points, spiral_points])

# Build KDTree
tree = KDTree(points)

# --- Example 1: Find nearest neighbors of one point
query_point = spiral_points[40]     # pick one spiral point
distances, indices = tree.query(query_point, k=10)  # 10 nearest neighbors
vicini = points[indices]
print("Nearest neighbors:", points[indices])

# --- Example 2: Extract points within a radius
radius = 1.0
indices_in_radius = tree.query_ball_point(query_point, r=radius)
neighbors = points[indices_in_radius]

# Plot everything
plt.figure(figsize=(10,10))
plt.scatter(vicini[:,0], vicini[:,1], color="lime", s=60, edgecolor="k", label="Nearest neighbors")
plt.scatter(points[:,0], points[:,1], s=20, alpha=0.5, label="All points")
plt.scatter(spiral_points[:,0], spiral_points[:,1], color="red", label="Spiral")
#plt.scatter(neighbors[:,0], neighbors[:,1], color="lime", s=60, edgecolor="k", label="Neighbors")
plt.scatter(*query_point, color="blue", s=100, marker="x", label="Query point")
plt.legend()
plt.title("KDTree neighbor search")
plt.show()

def vicini(points, size, bound=np.inf, start="b", neig=10):
    #work on points in the R phi representation to avoid infinite loops
    tree = KDTree(points)
    if start == "b":
        #start from the bottom
        partial = points[points[:, 1] < 0]
    elif start == "t":
        #start from the top
        partial = points[points[:, 1] > 0]
    else:
        print("Invalid start point. Use 'b' for bottom or 't' for top.")
        return None

    max = np.argmax(partial[:, 0])
    mean = np.mean(points[:, 0])
    query_point = points[max]  # pick one point

    visited = set()
    visited.add(tuple(query_point))  # store as tuple for hashing

    all_ind = []

    while query_point[0] > mean:
        distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)

        # mask within adaptive bound
        temp_bound = bound
        mask = distances <= temp_bound
        while not np.any(mask):
            temp_bound += 0.1
            distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)
            mask = distances <= temp_bound

        distances = distances[mask]
        indices   = indices[mask]

        # remove already visited points
        mask_new = [tuple(points[i]) not in visited for i in indices]
        distances = distances[mask_new]
        indices   = indices[mask_new]

        if indices.size == 0:
            print("no new neighbors - stopping")
            break

        # keep only CCW neighbors
        dphi = (points[indices,1] - query_point[1] + np.pi) % (2*np.pi) - np.pi
        mask_ccw = dphi > 0
        distances = distances[mask_ccw]
        indices   = indices[mask_ccw]

        if indices.size == 0:
            print("no counterclockwise neighbors - stopping")
            break

        # pick farthest CCW
        ind_dmax = np.argmax(distances)
        query_point = points[indices[ind_dmax]]
        visited.add(tuple(query_point))

        all_ind.append(indices[ind_dmax])

        #print("Next query point:", query_point)
    #remove duplicates
    all_ind = np.unique(all_ind)
    vic = points[all_ind]
    return vic


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


# --- Inputs ---
# image: 2D numpy array (grayscale)
# Provide center if known; else defaults to image center
def fit_spiral_with_deprojection(img, inclination_deg=15.0, center=None, PA_deg=None, threshold=None):
    #img = np.asarray(image)
    ny, nx = img.shape

    # 1) Choose/estimate center
    if center is None:
        x0, y0 = nx/2.0, ny/2.0
    else:
        x0, y0 = center

    # 2) Pick pixels belonging to the spiral (simple threshold or edges)
    if threshold is None:
        threshold = img.mean() + img.std()
    ys, xs = np.where(img > threshold)

    if len(xs) < 50:
        # Fall back to edges if thresholding is too sparse
        im8 = np.clip((img - img.min()) / (img.ptp() + 1e-9) * 255, 0, 255).astype(np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(im8, (5,5), 0), 50, 150)
        ys, xs = np.where(edges > 0)

    # Shift to center
    X = xs - x0
    Y = ys - y0

    # 3) Estimate PA (if not given) from 2D covariance of bright pixels
    #    PA = angle of major axis (in radians), measured from +x toward +y
    if PA_deg is None:
        pts = np.vstack([X, Y])  # shape (2, N)
        C = np.cov(pts)          # 2x2 covariance
        evals, evecs = np.linalg.eigh(C)
        major = evecs[:, np.argmax(evals)]
        PA = np.arctan2(major[1], major[0])
        PA_deg_est = np.degrees(PA)
    else:
        PA = np.radians(PA_deg)
        PA_deg_est = PA_deg

    # 4) Rotate points by -PA so major axis is horizontal
    c, s = np.cos(-PA), np.sin(-PA)
    Xr = c*X - s*Y
    Yr = s*X + c*Y

    # 5) Deproject: stretch the (projected) minor axis by 1/cos(i)
    i = np.radians(inclination_deg)
    deproj_factor = 1.0 / np.cos(i)  # ~1.0353 for 15°
    Yd = Yr * deproj_factor
    Xd = Xr

    # 6) Convert to polar in the deprojected (face-on) plane
    r = np.hypot(Xd, Yd)
    theta = np.arctan2(Yd, Xd)
    theta = np.unwrap(theta)

    # Optionally filter out tiny/huge radii (robustness)
    good = (r > np.percentile(r, 5)) & (r < np.percentile(r, 95))
    r_fit = r[good]
    theta_fit = theta[good]

    # 7) Fit logarithmic spiral: r = a * exp(b * theta)
    def log_spiral(theta_vals, a, b):
        return a * np.exp(b * theta_vals)

    # Initial guesses
    p0 = [np.median(r_fit[r_fit > 0]) if np.any(r_fit > 0) else 1.0, 0.1]
    popt, pcov = curve_fit(log_spiral, theta_fit, r_fit, p0=p0, maxfev=20000)
    a_fit, b_fit = popt

    # 8) Pitch angle (face-on)
    # tan(phi) = 1/b  ->  phi = arctan(1/b)
    phi = np.degrees(np.arctan(1.0 / b_fit))

    # 9) Create a smooth fitted curve (in deprojected plane) and map back to image frame
    tmin, tmax = theta_fit.min(), theta_fit.max()
    theta_s = np.linspace(tmin, tmax, 2000)
    r_s = log_spiral(theta_s, a_fit, b_fit)
    Xd_s = r_s * np.cos(theta_s)
    Yd_s = r_s * np.sin(theta_s)

    # Reproject to sky plane: undo deprojection, then rotate back, then shift to center
    Yr_s = Yd_s / deproj_factor
    Xr_s = Xd_s
    c2, s2 = np.cos(PA), np.sin(PA)
    Xs_s = c2*Xr_s + (-s2)*Yr_s
    Ys_s = s2*Xr_s +  c2*Yr_s
    xs_s = Xs_s + x0
    ys_s = Ys_s + y0

    result = {
        "a": a_fit,
        "b": b_fit,
        "pitch_angle_deg": phi,
        "PA_deg_used": PA_deg_est,
        "inclination_deg": inclination_deg,
        "center": (x0, y0),
        "theta_range": (float(tmin), float(tmax)),
        "curve_xy_image": (xs_s, ys_s),  # spiral back in the original image coords
        "mask_points": (xs, ys)          # points used for the fit (pre-rotation/deprojection)
    }
    return result

# ---------- Example of plotting ----------
folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

hdul = fits.open(name)
image_data = hdul[0].data
hdul.close()
image = image_data[0, 0, 0, :, :]  # select first frame

res = fit_spiral_with_deprojection(image, inclination_deg=15.0, center=None, PA_deg=None)

"""
xs_s, ys_s = res["curve_xy_image"]
#plt.imshow(image, cmap="gray", origin="lower")
#plt.plot(res["mask_points"][0], res["mask_points"][1], '.', ms=1, alpha=0.2)
plt.plot(xs_s, ys_s, '-', lw=2)
plt.title(f"Log Spiral fit: pitch={res['pitch_angle_deg']:.2f}°, PA={res['PA_deg_used']:.1f}°")
plt.show()
"""

# 1. Shift image to center
ny, nx = image.shape
x0, y0 = nx/2.0, ny/2.0
X_full = np.arange(nx) - x0
Y_full = np.arange(ny) - y0
X_grid, Y_grid = np.meshgrid(X_full, Y_full)

# 2. Rotate by -PA
PA = np.radians(res["PA_deg_used"])
c, s = np.cos(-PA), np.sin(-PA)
Xr_grid = c*X_grid - s*Y_grid
Yr_grid = s*X_grid + c*Y_grid

# 3. Deproject
i = np.radians(res["inclination_deg"])
deproj_factor = 1.0 / np.cos(i)
Yd_grid = Yr_grid * deproj_factor
Xd_grid = Xr_grid

# 4. Interpolate original image onto deprojected grid
#    Use map_coordinates for subpixel mapping
from scipy.ndimage import map_coordinates


# Coordinates for interpolation must be in (row, col) order
coords = np.array([Ys_back.ravel(), Xs_back.ravel()])

deproj_image = map_coordinates(image, coords, order=1, mode='reflect').reshape(ny, nx)

# 5. Plot
plt.figure(figsize=(8,8))
plt.imshow(deproj_image, cmap='inferno', origin='lower')
#plt.plot(Xd_s + x0, Yd_s + y0, 'c-', linewidth=2, label='Fitted spiral (deprojected)')
#plt.scatter(Xd, Yd, s=2, c='w', alpha=0.3, label='Detected pixels')
plt.legend()
plt.title("Deprojected Spiral Image")
plt.show()


dati = []
for i, arg in enumerate(sys.argv):
    if i>0:
        path = os.path.expanduser(arg)
        image = np.loadtxt(path, dtype=float)
        dati.append(image)

pixel_size = 320/image.shape[0]  # AU
hh = 3.5
allpeaks = []
styles = ['o', 's', '^', 'v', 'D', 'x', '*', 'p', 'h', '+', 'd']

plt.figure(figsize=(10, 10))
for j, image in enumerate(dati):
    peaks = []
    for i in range(image.shape[0]):
        peaks_in_one_trace, _ = find_peaks(image[i, :], height=hh)
        
        # Store peaks for each trace
        if peaks_in_one_trace.size > 0:
            for p in peaks_in_one_trace:
                peaks.append((i, p))
    allpeaks.append(peaks)

    # Select from image the peaks found
    rows, cols = zip(*peaks)  # unzip
    spiral = np.zeros_like(image)
    spiral[rows, cols] = image[rows, cols]
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    #rphi map
    rows = np.array(rows)
    cols = np.array(cols)
    ny, nx = image.shape
    x0, y0 = nx/2, ny/2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    #plot the R-phi map of the peaks
    plt.scatter(r, phi, s=1, alpha=0.5, marker=styles[j % len(styles)], label=f"Dataset {j+1}")

plt.title("R-$\phi$ Map of Peaks")
plt.xlabel("Radius (AU)")
plt.ylabel("$\phi$ (radians)")
plt.legend()
plt.grid()
#plt.savefig(outfile, bbox_inches="tight")
plt.show()