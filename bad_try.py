from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.signal import find_peaks


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
    min_r = np.min(points[:, 0])

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
                cand = [i for i in cand if ((r0 - points[i, 0]) >= 0 and (r0 - points[i, 0]) <= max_dr) or ((r0 - points[i, 0]) < 0 and (points[i, 0] - r0) <= max_dr/2)]

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


#wrong
def spiral_finder(image, hh, tt=0):
    peaks = []
    # Find peaks in the image data
    for i in range(image.shape[0]):
        peaks_in_one_trace, _ = find_peaks(image[:, i], height=hh, threshold=tt)
        
        # Store peaks for each trace
        if peaks_in_one_trace.size > 0:
            for p in peaks_in_one_trace:
                peaks.append((i, p))

    return np.array(peaks)


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


def sort_and_monotonic_smooth(data, smooth_window=15):
    sm_dt = []
    for sd_array in data:
        # Sort by second column (ascending)
        sorted_data = sd_array[sd_array[:, 1].argsort()]
        
        # Smooth the first column
        smoothed_first = uniform_filter1d(sorted_data[:, 0], size=smooth_window, mode='nearest')
        
        # Make the first column monotonically decreasing
        for i in range(1, len(smoothed_first)):
            if smoothed_first[i] > smoothed_first[i-1]:
                smoothed_first[i] = smoothed_first[i-1]
        
        sm_dt.append(np.column_stack((smoothed_first, sorted_data[:, 1])))
    
    return sm_dt


def extrapolate_phi_local(points, r_cut, kind='linear', dr=None):
    """
    Extrapolate phi(r) for r > r_cut using only data at r <= r_cut.

    points : (N,2) array of (r, phi) with phi in radians
    r_cut  : float, cutoff radius
    kind   : 'linear', 'quadratic', 'cubic', ...
    dr     : if set, use only points with r in [r_cut - dr, r_cut]
             (ignored if last_n is provided)

    Returns: (N,2) array with phi replaced for r > r_cut
    """
    pts = np.asarray(points, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N,2) array of (r, phi)")

    # sort by radius
    order = np.argsort(pts[:, 0])
    r = pts[order, 0]
    phi = pts[order, 1]

    # base mask: r <= r_cut
    base = r <= r_cut
    if not np.any(base):
        raise ValueError("No points at or below r_cut")

    if dr is not None:
        idx_subset = np.where((r >= (r_cut - dr)) & (r <= r_cut))[0]
    else:
        idx_subset = np.where(base)[0]
    
    # need enough points for the chosen 'kind'
    min_pts = {'linear': 2, 'nearest': 1, 'zero': 1,
               'slinear': 2, 'quadratic': 3, 'cubic': 4}.get(kind, 2)
    if idx_subset.size < min_pts:
        raise ValueError(f"Need at least {min_pts} points for kind='{kind}', "
                         f"but got {idx_subset.size}")

    r_sub = r[idx_subset]
    phi_sub = phi[idx_subset]

    # interp1d needs strictly increasing x: ensure uniqueness
    r_sub_unique, unique_idx = np.unique(r_sub, return_index=True)
    phi_sub_unique = phi_sub[unique_idx]
    if r_sub_unique.size < min_pts:
        raise ValueError("After removing duplicate radii, not enough points remain.")

    f = interp1d(r_sub_unique, phi_sub_unique, kind=kind,
                 fill_value='extrapolate', assume_sorted=True)

    # extrapolate for r > r_cut
    mask_out = r > r_cut
    phi[mask_out] = f(r[mask_out])

    # place back in original order
    out = np.empty_like(pts)
    out[order, 0] = r
    out[order, 1] = phi
    return out



def bad)track():
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

        if r_sorted[i] <= avg or dif <= (lim/px_size):
            filtered_peaks.append(sorted_peaks[i])
        else:
            r_sorted[i] = r_sorted[i-1]
            x, y = rphi_to_xy(r_sorted[i-1], phi_sorted[i], image_size)
            sorted_peaks[i] = np.array(y,x)
    
    return np.array(filtered_peaks)





#############
#===========================================================
############# function to sort and smooth the distances
def filter_bads(data, m, mm):
    filtered_indices = np.ones(len(data), dtype=bool)
    phi = data[:,1]
    R = data[:,0]
    # Define angle window in radians 
    angle_min = np.deg2rad(m)
    angle_max = np.deg2rad(mm)

    # Iteratively apply the filter in 5 degree windows within [angle_min, angle_max]
    window_deg = 5
    for start_deg in range(int(np.rad2deg(angle_min)), int(np.rad2deg(angle_max)), window_deg):
        win_min = np.deg2rad(start_deg)
        win_max = np.deg2rad(start_deg + window_deg)
        angle_mask = (phi >= win_min) & (phi < win_max)
        subset_R = R[angle_mask]
        mean_val = np.mean(subset_R) if subset_R.size > 0 else 0

        # Keep only radii over the average in this angle window
        if subset_R.size > 0:
            filtered_indices[angle_mask] = subset_R > mean_val

    return data[filtered_indices]
    

#############
#===========================================================
############# function to fill gaps
def fill_phi_gaps(data, max_gap_deg=5):
    # Sort data by phi
    data_sorted = data[np.argsort(data[:, 1])]
    R = data_sorted[:, 0]
    phi = data_sorted[:, 1]
    filled_R = []
    filled_phi = []

    for i in range(len(phi) - 1):
        filled_R.append(R[i])
        filled_phi.append(phi[i])
        gap = np.rad2deg(phi[i+1] - phi[i])
        if gap > max_gap_deg:
            # Number of points to fill (1 per deg, excluding endpoints)
            n_fill = int(gap) - 1
            if n_fill > 0:
                phi_fill = np.linspace(phi[i] + np.deg2rad(1), phi[i+1] - np.deg2rad(1), n_fill)
                R_fill = np.linspace(R[i], R[i+1], n_fill + 2)[1:-1]  # exclude endpoints
                filled_R.extend(R_fill)
                filled_phi.extend(phi_fill)

    # Add last point
    filled_R.append(R[-1])
    filled_phi.append(phi[-1])
    return np.column_stack((filled_R, filled_phi))


#############
#===========================================================
############# function to extrapolate
def extrapolate_phi_in(points, r_in, r_out, in_lim = 20, out_lim=100, type="log"):
    points = points.copy()
    r = points[:, 0]
    phi = points[:, 1]

    # Fit linear model phi(r) only for r <= r_cut
    mask = (r <= r_in) | (r >= r_out)
    if in_lim <= r_in:
        mask &= r >= in_lim
    elif out_lim >= r_out:
        mask &= r <= out_lim

    if np.sum(mask) < 2:
        raise ValueError("Need at least 2 points below r_cut for linear extrapolation")

        # Fit chosen spiral model
    if type == "log":
        X = np.log(r[mask])
    elif type == "arch":
        X = r[mask]
    else:
        raise ValueError("type must be 'log' or 'arch'")

    coeffs = np.polyfit(X, phi[mask], 1)
    a, b = coeffs

    # Extrapolate inside (r_in, r_out)
    mask_out = (r > r_in) & (r < r_out)
    if type == "log":
        phi[mask_out] = a * np.log(r[mask_out]) + b
    else:  # "arch"
        phi[mask_out] = a * r[mask_out] + b

    points[:, 1] = phi
    return points

#############
#===========================================================
############# function for bad practice
def fix_phi_monotonic(datasets, max_iter=1000):
    datasets = [ds.copy() for ds in datasets]  # work on copies

    changed = True
    iterations = 0

    while changed and iterations < max_iter:
        changed = False
        iterations += 1

        for i in range(1, len(datasets)):
            prev = datasets[i - 1]
            curr = datasets[i]

            for j in range(len(curr)):
                if curr[j, 1] < prev[j, 1]:  # check phi
                    # swap phi values
                    prev[j, 1], curr[j, 1] = curr[j, 1], prev[j, 1]
                    changed = True

    return datasets


#######from interpoling.py
    """
    input_2 = base + ratio + "/" + subdir + "/" + str(i) + "_i" + arm + ".txt"
    if os.path.exists(input_2):
        data2 = read_single(input_2)
        data = np.vstack((data2, data))
    """
    #per sim 03 bot - raccordo
    #data = extrapolate_phi_in(data, 40,50)


#interp_r = int_all(all, n=300, i=1)
#plot_all_phi_r(interp_r, "interp", dt)

#all[1] = extrapolate_phi_in(all[1], 30, 48,30,53)
#all[2] = extrapolate_phi_in(all[2], 50, 75, 50)
"""
for i, data in enumerate(all):
    #data = data[data[:, 0] >= 40]
    #data = fill_phi_gaps(data)
    data = filter_bads(data, -10, 30)
    #data = filter_bads(data, 100, 135)

    if i*dt == 10:
        #data = extrapolate_phi_in(data, 77, 100)
        #data = extrapolate_phi_in(data, 45, 60)
        data = extrapolate_phi_in(data, 30, 50)

    if i*dt == 0:
        data = extrapolate_phi_in(data, 30, 50)

    elif i*dt == 5:
        data = extrapolate_phi_in(data, 20, 23)
    #int_data = interp(data, np.min(data[:,0]), np.max(data[:,0]), 200, 0)

    all[i] = data
#"""


#######from mod_img
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



"""for fitting velocities an plotting
outfile = "~/thesis/img_ts/"+ratio+"_vel_fits"
base = "~/thesis/Spiral_pattern/"+ratio+"/results/"
fine = "_"+arm+"_int_"+str(dt)+"_vel.txt"

Rs, vvs = [], []

file = os.path.expanduser(base+"sim"+fine)
R, vels = read_R_data_file(file)
Rs.append(R)
vvs.append(vels)

file = os.path.expanduser(base+"mc"+fine)
R, vels = read_R_data_file(file)
Rs.append(R)
vvs.append(vels)

...

for i, v in enumerate(vvs):
    axs[i].plot(Rs[i], vvs[i][0], '-', lw=linewidth, label="Computed velocity")
    
"""