from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from astropy.constants import G, au, M_sun
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob

#############
#===========================================================
############# function to read single file
def read_single(input_file):
    # Read array from input file (two columns: R and phi, use 'nan' for missing phi)
    data = np.loadtxt(input_file, dtype=float)
    R = data[:, 0]
    phi = data[:,1]

    # Indices of valid (non-nan) phi values
    valid = ~np.isnan(phi)
    R_valid = R[valid]
    phi_valid = phi[valid]
    data_valid = np.column_stack((R_valid, phi_valid))
    print("size: ",data_valid.size)

    return data_valid


#############
#===========================================================
############# function to read multiple file
def read_mult(dire, arm, st=1):
    all_data = []
    for i in range(0, 11, st):
        input_file = dire + str(i) + "_" + arm + ".txt"
        if os.path.exists(input_file):
            data = read_single(input_file)
            all_data.append(data)
        else:
            print(f"File {input_file} not found, skipping.")

    print("len ",len(all_data))
    return all_data


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
    continuous = np.linspace(min, max, n)

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
    dphi_dt = (phi_curr - phi_prev) / (dt*len(int_dt))
    velocities.append(np.column_stack((R, dphi_dt)))

    return velocities


#############
#===========================================================
############# function to plot all the phi(R) maps
def plot_all_phi_r(int_dt, title, dt):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 0], data[:, 1], label=f'Year {dt*i}')
    plt.xlabel('R (AU)')
    plt.ylabel('$\phi$ (radians)')
    plt.title(title)

    # set y ticks in multiples of π/2
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid(True)
    plt.show()

#############
#===========================================================
############# function to plot all the R(phi) maps
def plot_all_r_phi(int_dt, title, dt):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 1], data[:, 0], label=f'Year {i*dt}')
    plt.ylabel('R (AU)')
    plt.xlabel('$\phi$ (radians)')
    plt.title(title)

    # set y ticks in multiples of π/2
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid(True)
    plt.show()


#############
#===========================================================
############# function to plot the velocities
def plot_vel(int_dt, dt):
    #convert units for consistency
    radii = np.linspace(np.min(int_dt[0][:,0]), np.max(int_dt[0][:,0]), 100) * u.au
    omegaK = np.sqrt(G*M_sun/radii**3)
    # Convert to rad/year
    omegaK_year = omegaK.to(1/u.yr)   # "1/u.yr" means frequency in cycles per year (rad/year)
    #print(omegaK_year)

    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        if i*dt != 10:
            label=fr'Year {i*dt} $\rightarrow$ {dt*(i+1)}'
        else:
            label=fr'Year 0 $\rightarrow$ 10'
        plt.plot(data[:, 0], data[:, 1], label=label)
    plt.plot(radii, omegaK_year, '.', label="Keplerian angular velocity")
    plt.xlabel('R (AU)')
    plt.ylabel(r"$\frac{d\phi}{dt} = \Omega(R)$ (radians/year)")
    plt.title("Angular pattern velocity")

    plt.legend()
    plt.grid(True)
    plt.show()


#############
#===========================================================
############# function to save the velocities
def save_vel(velocities):
    file = input("ratio/filename to svae> ")
    filename = "Spiral_pattern/"+file

    # Assume all velocities arrays have the same radii
    radii = velocities[0][:, 0]
    # Stack all velocity columns together
    vel_columns = [v[:, 1] for v in velocities]
    data_to_save = np.column_stack([radii] + vel_columns)
    # Save to txt file with header
    header = "".join([f"\tomega_{i}" for i in range(len(velocities))])
    np.savetxt(filename, data_to_save, header=header, fmt="%.8f", delimiter="\t")


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
def extrapolate_phi_in(points, r_in, r_out):
    points = points.copy()
    r = points[:, 0]
    phi = points[:, 1]

    # Fit linear model phi(r) only for r <= r_cut
    mask = (r <= r_in) | (r >= r_out)
    if np.sum(mask) < 2:
        raise ValueError("Need at least 2 points below r_cut for linear extrapolation")

    coeffs = np.polyfit(r[mask], phi[mask], 1)  # slope, intercept
    slope, intercept = coeffs

    # Replace phi for r > r_cut
    mask_out = (r > r_in) & (r < r_out)
    phi[mask_out] = slope * r[mask_out] + intercept

    points[:, 1] = phi
    return points