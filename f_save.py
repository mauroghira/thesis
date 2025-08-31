import numpy as np
from f_gen import *
import csv

#############
#===========================================================
############# function to save the datas afteer smoothing
def save_rphi_all(velocities, filename):
    radii = velocities[0][:, 0]
    # Stack all velocity columns together
    vel_columns = [v[:, 1] for v in velocities]
    data_to_save = np.column_stack([radii] + vel_columns)
    # Save to txt file with header
    header = "".join([f"\tphi_year_{i}" for i in range(len(velocities))])
    np.savetxt(filename, data_to_save, header=header, fmt="%.8f", delimiter="\t")


def save_vel(velocities, filename):
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
############# function to save the spiral arm from the image at fixed year

def save_rphi(points, file, phimax, scale, size):
    #scaling radius
    r, phi = xy_to_rphi(points[:,0], points[:,1], size)
    r = r * scale

    #if starting from the top, make the angle monotonous
    if phimax > 180:
        phi = (phi + 2*np.pi) % (2*np.pi)

    data = np.column_stack((r,phi))
    with open(file, 'w') as f:
        csv.writer(f, delimiter=' ').writerows(data)