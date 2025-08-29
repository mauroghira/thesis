import numpy as np

#############
#===========================================================
############# function to convert coords

def xy_to_rphi(rows, cols, size):
    # Convert pixel coordinates to R-phi coordinates
    ny, nx = size-1, size-1
    x0, y0 = nx / 2, ny / 2
    x = cols - x0
    y = rows - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return r, phi

def rphi_to_xy(r, phi, size):
    x = r * np.cos(phi) + (size-1) / 2
    y = r * np.sin(phi) + (size-1) / 2

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