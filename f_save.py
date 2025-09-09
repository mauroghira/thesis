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


def save_vel(r, vm, vstd, filename):
    """
    Save velocity results in columns:
    R, vm1 ... vmN, vstd1 ... vstdN
    """
    # Ensure vm and vstd are 2D with shape (N_points, N_curves)
    vm = np.atleast_2d(vm)
    vstd = np.atleast_2d(vstd)

    # If they came as (N_curves, N_points), transpose
    if vm.shape[0] != len(r):
        vm = vm.T
    if vstd.shape[0] != len(r):
        vstd = vstd.T

    # Stack all columns: R | vm... | vstd...
    data_to_save = np.column_stack([r, vm, vstd])

    # Build header
    n_cols = vm.shape[1]
    header_vm = "\t".join([f"vm{i}" for i in range(n_cols)])
    header_vstd = "\t".join([f"vstd{i}" for i in range(n_cols)])
    header = "R\t" + header_vm + "\t" + header_vstd

    # Save
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