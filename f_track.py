import numpy as np

from scipy.ndimage import maximum_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from f_gen import *

#############
#===========================================================
############# function to find the spiral arms in the image
def find_2d_peaks(image, threshold=0, max=5, window=3):
    # Apply a maximum filter to find local maxima
    neighborhood = np.ones((window, window))
    local_max = (image == maximum_filter(image, footprint=neighborhood))
    detected_peaks = np.argwhere(local_max & (image > threshold) & (image < max))
    return detected_peaks


#############
#===========================================================
############# function to select an arm
def filter_peaks_by_rphi(peaks, image_size, px_size, r_min=0, r_max=100, phi_min=-90, phi_max=90):
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

    return partial


#############
#===========================================================
############# function to process data before selecting
def modify_r_by_phi_extremes(xy_coords, image_size, phi_min_deg, phi_max_deg, mode='out'):
    # Convert xy to polar coordinates
    rows, cols = xy_coords[:, 0], xy_coords[:, 1]
    r, phi = xy_to_rphi(rows, cols, image_size)

    # Define angular extremes in radians
    phi_min = np.deg2rad(phi_min_deg)
    phi_max = np.deg2rad(phi_max_deg)
    if phi_max > np.pi:
        phi = (phi + 2*np.pi) % (2*np.pi)

    # Sort by phi
    sort_idx = np.argsort(phi)
    r_sorted = r[sort_idx]
    phi_sorted = phi[sort_idx]
    coords_sorted = np.column_stack((r_sorted, phi_sorted))

    subset_mask = (phi_sorted >= phi_min) & (phi_sorted <= phi_max)

    # Modify r in the subset
    subset_r = r_sorted[subset_mask]
    if subset_r.size > 0:
        avg_r = np.mean(subset_r)

        if mode == 'out':
            keep_mask = subset_r > avg_r
            # Remove points in the angular window that are not > avg_r
            remove_mask = subset_mask.copy()
            remove_mask[subset_mask] = ~keep_mask
            r_sorted = r_sorted[~remove_mask]
            phi_sorted = phi_sorted[~remove_mask]

        elif mode == 'in':
            keep_mask = subset_r < avg_r
            # Remove points in the angular window that are not < avg_r
            remove_mask = subset_mask.copy()
            remove_mask[subset_mask] = ~keep_mask
            r_sorted = r_sorted[~remove_mask]
            phi_sorted = phi_sorted[~remove_mask]

        elif mode == 'fit':
            # Interpolate r using a logarithmic spiral formula: r = a * exp(b * phi)
            # Fit spiral parameters to the subset
            if subset_r.size < 5:
                # Generate 5 equally spaced phi values between phi_min and phi_max
                phi_new = np.linspace(phi_min, phi_max, 7)
                # Fit spiral parameters to the available subset
                try:
                    popt, _ = curve_fit(spiral, phi_sorted[subset_mask], subset_r, maxfev=10000)
                    interp_r = spiral(phi_new, *popt)
                except Exception:
                    interp_r = np.linspace(r[0], r[-1], 7)
                # Replace subset with new points
                r_sorted = np.concatenate([r_sorted[~subset_mask], interp_r])
                phi_sorted = np.concatenate([phi_sorted[~subset_mask], phi_new])
            else:
                interp_r = fit(subset_r, phi_sorted[subset_mask])  
                r_sorted[subset_mask] = interp_r

        elif mode == 'del':
            # Remove points in the angular window
            r_sorted = r_sorted[~subset_mask]
            phi_sorted = phi_sorted[~subset_mask]
        
        elif mode == 'avg':
            # Local averaging within ±5 degrees
            window = np.deg2rad(5)
            new_r = np.copy(r_sorted)

            for i, (phi_val, r_val) in enumerate(zip(phi_sorted, r_sorted)):
                if subset_mask[i]:
                    # Find neighbors within ±5° of current phi
                    neighbor_mask = (phi_sorted >= phi_val - window) & (phi_sorted <= phi_val + window)
                    neighbor_r = r_sorted[neighbor_mask]
                    if neighbor_r.size > 0:
                        new_r[i] = np.mean(neighbor_r)

            r_sorted = new_r

    x,y = rphi_to_xy(r_sorted, phi_sorted, image_size)
    return np.column_stack((y, x))

#############
#===========================================================
############# function for local fit of the spiral
def fit(r, phi):
    try:
        popt, _ = curve_fit(spiral, phi, r, maxfev=10000)
        interp_r = spiral(phi, *popt)
    except Exception:
        # Fallback to linear interpolation if fitting fails
        interp_r = np.linspace(r[0], r[-1], len(r))
    return interp_r