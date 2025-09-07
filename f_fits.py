import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from f_gen import *

#############
#===========================================================
############# function to fit - used in fit_phi_r.py
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
############# function to interpolate in common range fitted data - used in fit_phi_r.py
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
############# function to fits velocity
def fit_vel(r, omega, err, model):
    popt, pcov = curve_fit(model, r, omega, maxfev=10000, sigma=err, absolute_sigma=True)

    rsq = r_squared(omega, model(r, *popt))
    chi2, chired, p = chi_sq(r, omega, model, popt, err)

    return rsq, chi2, chired, p , popt