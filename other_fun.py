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
def read_mult(dire, arm):
    all_data = []
    for i in range(11):
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
def int_all(all_data, n):
    min, max = select_extremes(all_data, 0)
    int_dt=[]
    for data in all_data:
        int_dt.append(interp(data, min, max, n))

    return int_dt

#############
#===========================================================
############# function to interpolate the single dataset in phi
def interp(data, min, max, n):
    R = data[:,0]
    phi = data[:,1]
    # Create a more continuous R array for interpolation
    R_continuous = np.linspace(min, max, n)

    # Interpolate using linear method within the range of R
    f_interp = interp1d(R, phi, kind='linear', bounds_error=False, fill_value="extrapolate")
    phi_continuous = f_interp(R_continuous)

    return np.column_stack((R_continuous, phi_continuous))


#############
#===========================================================
############# function to smooth the data 
def smooth(int_dt, i=1, window_size=15):
    #i=0 tosm R, 1 for phi
    smoothed = []
    for data in int_dt:
        # uniform_filter1d applies a moving average with reflection at edges
        smooth = uniform_filter1d(data[:,i%2], size=window_size, mode='nearest')
        
        if i%2 == 0:
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
    for i in range(1, len(int_dt)):
        R = int_dt[i][:, 0]
        phi_prev = int_dt[i-1][:, 1]
        phi_curr = int_dt[i][:, 1]
        # Compute time derivative (finite difference)
        dphi_dt = (phi_curr - phi_prev) / dt
        velocities.append(np.column_stack((R, dphi_dt)))
    return velocities


#############
#===========================================================
############# function to plot all the phi(R) maps
def plot_all_phi_r(int_dt, title):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 0], data[:, 1], label=f'Year {i}')
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
def plot_all_r_phi(int_dt, title):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 1], data[:, 0], label=f'Year {i}')
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
def plot_vel(int_dt):
    #convert units for consistency
    radii = np.linspace(np.min(int_dt[0][:,0]), np.max(int_dt[0][:,0]), 100) * u.au
    omegaK = np.sqrt(G*M_sun/radii**3)
    # Convert to rad/year
    omegaK_year = omegaK.to(1/u.yr)   # "1/u.yr" means frequency in cycles per year (rad/year)
    #print(omegaK_year)

    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 0], data[:, 1], label=fr'Year {i} $\rightarrow$ {i+1}')
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