from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
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
    input_files = glob.glob(dire+'*'+arm+".txt")
    print(input_files)
    all_data = []
    for input_file in input_files:
        data = read_single(input_file)
        all_data.append(data)
    print("len ",len(all_data))
    return all_data


#############
#===========================================================
############# function to select the interpolation extremes
def select_extremes(all_data):
    min_values = [np.min(data[:, 0]) for data in all_data if data.size > 0]
    max_values = [np.max(data[:, 0]) for data in all_data if data.size > 0]
    min_extreme = max(min_values)
    max_extreme = min(max_values)
    return min_extreme, max_extreme


#############
#===========================================================
############# function to interpolate the whole dataset
def int_all(all_data):
    min, max = select_extremes(all_data)
    int_dt=[]
    for data in all_data:
        int_dt.append(interp(data, min, max))

    return int_dt

#############
#===========================================================
############# function to interpolate the single dataset
def interp(data, min, max):
    R = data[:,0]
    phi = data[:,1]
    # Create a more continuous R array for interpolation
    R_continuous = np.linspace(min, max, 1000)

    # Interpolate using linear method within the range of R
    f_interp = interp1d(R, phi, kind='linear', bounds_error=False, fill_value="extrapolate")
    phi_continuous = f_interp(R_continuous)

    return np.column_stack((R_continuous, phi_continuous))


#############
#===========================================================
############# function to smooth the data
def smooth(int_dt, window_size=15):
    smoothed = []
    for data in int_dt:
        R = data[:, 0]
        phi = data[:, 1]
        # uniform_filter1d applies a moving average with reflection at edges
        phi_smooth = uniform_filter1d(phi, size=window_size, mode='nearest')
        
        smoothed.append(np.column_stack((R, phi_smooth)))
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
############# function to plot all the datasets
def plot_all(int_dt, title):
    #plt.figure(figsize=(9, 9))
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
############# function to plot the velocities
def plot_vel(int_dt):
    #plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 0], data[:, 1], label=fr'Year {i} $\rightarrow$ {i+1}')
    plt.xlabel('R (AU)')
    plt.ylabel(r"$\frac{d\phi}{dt} = \Omega(R)$ (radians/year)")
    plt.title("Angular pattern velocity")

    plt.legend()
    plt.grid(True)
    plt.show()