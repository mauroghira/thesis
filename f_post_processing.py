import numpy as np

from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from f_gen import *

#############
#===========================================================
############# function to fot
def fit_models(phi, r, candidates):
    for msg, model, (x, y, mode) in candidates:
        try:
            popt, _ = curve_fit(model, x, y, maxfev=10000)
            print(msg)
            Rquad = r_squared(y, model(x, *popt))
            print("R squared = ", Rquad)
            return popt, mode, model
        except Exception as e:
            continue

    print("unable to fit")
    return None, None, None

#############
#===========================================================
############# function to interpolate in common range fitted data

def resample_rphi(f_rphi, popt, r_min, r_max, phi_range, N=500):
    """
    Resample fitted r(phi) into equally spaced radii.

    Parameters
    ----------
    f_rphi : callable
        The fitted model function r(phi, *popt).
    popt : array
        Optimal parameters from curve_fit.
    r_min, r_max : float
        The radius interval to resample.
    N : int
        Number of equally spaced points in [r_min, r_max].
    phi_range : tuple
        Range of phi to sample for building the inverse interpolator.

    Returns
    -------
    r_new : ndarray
        Equally spaced radii in [r_min, r_max].
    phi_new : ndarray
        Corresponding phi values such that r(phi_new) ≈ r_new.
    """
    # Sample r(phi) on a fine phi-grid
    phi_grid = np.linspace(phi_range[0], phi_range[1], 5000)
    r_grid = f_rphi(phi_grid, *popt)

    # Build inverse interpolator φ(r)
    phi_of_r = interp1d(r_grid, phi_grid, kind="linear", bounds_error=False)

    # Equally spaced radii
    r_new = np.linspace(r_min, r_max, N)

    # Get corresponding φ
    phi_new = phi_of_r(r_new)

    return r_new, phi_new


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
def interp(data, m, MM, n, i=0):
    R = data[:,0]
    phi = data[:,1]
    # Create a more continuous R array for interpolation
    continuous = np.linspace(m, MM, n)

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
