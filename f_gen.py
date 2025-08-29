import numpy as np

from astropy.constants import G, au, M_sun
import astropy.units as u

#############
#===========================================================
############# function to convert coords

def xy_to_rphi(rows, cols, size, px=1):
    #px=0 if already centered
    # Convert pixel coordinates to R-phi coordinates
    ny, nx = (size-1)*px, (size-1)*px
    x0, y0 = nx / 2, ny / 2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi

def rphi_to_xy(r, phi, size, px=1):
    x = r * np.cos(phi) + (size-1)*px / 2
    y = r * np.sin(phi) + (size-1)*px / 2

    return x, y


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
############# function to select the interpolation extremes
def select_extremes(all_data, i):
    min_values = [np.min(data[:, i]) for data in all_data if data.size > i]
    max_values = [np.max(data[:, i]) for data in all_data if data.size > i]
    min_extreme = max(min_values)
    max_extreme = min(max_values)
    return min_extreme, max_extreme


#############
#===========================================================
############# function to compute the velocity
def compute_velocity(int_dt, dt=1.0):
    # Assumes int_dt is a list of arrays [R, phi] for each time step (year)
    velocities = []
    R = int_dt[0][:, 0]
    for i in range(1, len(int_dt)):
        phi_prev = int_dt[i-1][:, 1]
        phi_curr = int_dt[i][:, 1]
        # Compute time derivative (finite difference)
        dphi_dt = (phi_curr - phi_prev) / dt
        velocities.append(np.column_stack((R, dphi_dt)))
    
    #add velocity 0-10
    phi_prev = int_dt[0][:, 1]
    dphi_dt = (phi_curr - phi_prev) / (dt*(len(int_dt)-1))
    velocities.append(np.column_stack((R, dphi_dt)))

    return velocities

def kepler(int_dt):
    #convert units for consistency
    radii = np.linspace(np.min(int_dt), np.max(int_dt), 100) * u.au
    omegaK = np.sqrt(G*M_sun/radii**3)
    # Convert to rad/year
    omegaK_year = omegaK.to(1/u.yr)   # "1/u.yr" means frequency in cycles per year (rad/year)

    return radii, omegaK_year


#############
#===========================================================
############# function to fit the spirals
def spiral(phi, a, b):
    return a * np.exp(b * phi)

def logspir(r, a, b):
    return a * np.log(r) + b

#ok for r(phi) and phi(r) too
def archspir(r, a, b):
    return a + b * r

def r_squared(actual, predicted):
    residuals = actual - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum( (actual - np.mean(actual))**2 )
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared