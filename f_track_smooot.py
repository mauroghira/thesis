import numpy as np

from scipy.ndimage import maximum_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from f_gen import *

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
    """
    Given an array of xy coordinates, convert to polar (r, phi), sort by phi,
    and modify r in the subset defined by angular extremes.
    mode: 'higher', 'lower', or 'avg' to keep only higher/lower than avg or set to avg.
    Returns: array of (r, phi) sorted by phi, with r modified in the subset.
    """
    # Convert xy to polar coordinates
    rows, cols = xy_coords[:, 0], xy_coords[:, 1]
    r, phi = xy_to_rphi(rows, cols, image_size)

    # Sort by phi
    sort_idx = np.argsort(phi)
    r_sorted = r[sort_idx]
    phi_sorted = phi[sort_idx]
    coords_sorted = np.column_stack((r_sorted, phi_sorted))

    # Define angular extremes in radians
    phi_min = np.deg2rad(phi_min_deg)
    phi_max = np.deg2rad(phi_max_deg)
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

        elif mode == 'avg':
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

    x,y = rphi_to_xy(r_sorted, phi_sorted, image_size)
    return np.column_stack((y, x))


#############
#===========================================================
############# function for local fit of the spiral
def spiral(phi, a, b):
    return a * np.exp(b * phi)

def fit(r, phi):
    try:
        popt, _ = curve_fit(spiral, phi, r, maxfev=10000)
        interp_r = spiral(phi, *popt)
    except Exception:
        # Fallback to linear interpolation if fitting fails
        interp_r = np.linspace(r[0], r[-1], len(r))
    return interp_r

#############
#===========================================================
############# function to interpolate the whole dataset in phi
def int_all(all_data, n, i=0):
    i = i%2
    min, max = select_extremes(all_data, i)
    int_dt=[]
    for data in all_data:
        int_dt.append(interp(data, min, max, n, i))

    return int_dt

#############
#===========================================================
############# function to interpolate the single dataset in phi
def interp(data, min, max, n, i=0):
    R = data[:,0]
    phi = data[:,1]
    # Create a more continuous R array for interpolation
    continuous = np.linspace(min+1, max, n)

    # Interpolate using linear method within the range of R
    if i == 0:
        f_interp = interp1d(R, phi, kind='linear', bounds_error=False, fill_value="extrapolate")
        phi_continuous = f_interp(continuous)
        return np.column_stack((continuous, phi_continuous))

    else:
        f_interp = interp1d(phi, R, kind='linear', bounds_error=False, fill_value="extrapolate")
        R_continuous = f_interp(continuous)
        return np.column_stack((R_continuous, continuous))


#############
#===========================================================
############# function to smooth the data 
def smooth(int_dt, i=1, window_size=15):
    i = i%2
    #i=0 to sm R, 1 for phi
    smoothed = []
    for data in int_dt:
        # uniform_filter1d applies a moving average with reflection at edges
        smooth = uniform_filter1d(data[:,i], size=window_size, mode='nearest')

        if i == 0:
            smoothed.append(np.column_stack((smooth, data[:,1])))
        else:
            smoothed.append(np.column_stack((data[:,0], smooth)))

    return smoothed



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
def extrapolate_phi_in(points, r_in, r_out, in_lim = 20, out_lim=100):
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

    #"""
    coeffs = np.polyfit(r[mask], phi[mask], 1)  # slope, intercept
    slope, intercept = coeffs

    # Replace phi for r > r_cut
    mask_out = (r > r_in) & (r < r_out)
    phi[mask_out] = slope * r[mask_out] + intercept
    #"""

    points[:, 1] = phi
    return points