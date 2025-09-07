import numpy as np

from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

from f_gen import *

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

    if i==0:
        # --- Sort data ---
        sort_idx = np.argsort(R)
        R, phi = R[sort_idx], phi[sort_idx]

        # --- Average phi for duplicate R ---
        unique_R, inverse = np.unique(R, return_inverse=True)
        phi_avg = np.array([phi[inverse == k].mean() for k in range(len(unique_R))])

        # --- Clean NaN/Inf if needed ---
        mask = np.isfinite(unique_R) & np.isfinite(phi_avg)
        R, phi = unique_R[mask], phi_avg[mask]

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
############# function to wrap everything
def monte_carlo_velocities(all_data, sigma_R=5, n=270, window_size=15, dt=10, n_mc=100):
    """
    Propagating errors on R using Monte Carlo
    all_data: list of array (R, phi)
    sigma_R: error on R, array or constant (in AU)
    n: points per interpolation
    window_size: window for smoothing
    dt: time interval
    n_mc: number of MC simulations
    """
    velocities_mc = []

    for mc in range(n_mc):
        perturbed = []
        for dataset in all_data:
            R = dataset[:,0]
            phi = dataset[:,1]
            # perturba R
            R_pert = R + np.random.normal(0, sigma_R, size=R.shape)
            perturbed.append(np.column_stack((R_pert, phi)))

        # interpolating e smoothing
        interp_data = int_all(perturbed, n=n, i=0)
        smooth_data = smooth(interp_data, i=1, window_size=window_size)

        # compute velocity
        vel = compute_velocity(smooth_data, dt)
        velocities_mc.append([v[:,1] for v in vel])  # drop the R coord

    velocities_mc = np.array(velocities_mc)  # shape (n_mc, n_times-1, n_points)

    # mean and std
    v_mean = velocities_mc.mean(axis=0)
    v_std  = velocities_mc.std(axis=0)

    # R common (after interp)
    R_common = interp_data[0][:,0]

    return R_common, v_mean, v_std