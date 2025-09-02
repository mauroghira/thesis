import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *
from f_post_processing import fit_vel

if len(sys.argv) != 3:
    print("Usage: python plotting.py ratio file")
    sys.exit(1)

ratio = sys.argv[1]
infile = sys.argv[2]
inizio, dt = infile.rsplit("_", 1)
dt = int(dt)
base = "~/thesis/Spiral_pattern/"+ratio

images = []
for i in range(0,11,dt):

    if "sim" in infile:
        input_file = os.path.expanduser(base+"/img_"+ratio+"_"+str(i)+".pix")
        
        if os.path.exists(input_file):
            image = np.loadtxt(input_file, dtype=float)
            images.append(image)
        else:
            print(f"File {input_file} not found, skipping.")
        
        label = "log column density [g/Cm⁻²]"
        pixel_size = 320/544  # AU
        shape = 544
        vmin=None

    else:
        input_file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/inc_0/y"+str(i)+"/data_1300/RT.fits.gz")
    
        if os.path.exists(input_file):
            hdul = fits.open(input_file)
            image_data = hdul[0].data
            hdul.close()
            image = image_data[0, 0, 0, :, :]  # select first frame
            pixel_size = 320/512  # AU
            rot = np.rot90(image, k=1)
            diff = mod_img(rot, pixel_size)
            images.append(diff)
        else:
            print(f"File {input_file} not found, skipping.")

        label = "Flux [W/(m² pixel)]"
        shape = 512
        vmin = -1e-21


extent = [-(shape-1) * pixel_size/2, (shape-1) * pixel_size/2, -(shape-1) * pixel_size/2, (shape-1) * pixel_size/2]

file = os.path.expanduser(base+"/results/"+infile+"_phi.txt")
R, phis = read_R_data_file(file)

for i, phi in enumerate(phis):
    x, y = rphi_to_xy(R, phi, 0)
    out = os.path.expanduser("~/thesis/img_ts/"+ratio+"_"+inizio+"_"+str(i*dt)+"_phi.jpg")
    spiral_plot(x,y, images[i], label, extent, out, vmin)

file = os.path.expanduser(base+"/results/"+infile+"_vel.txt")

R, vels = read_R_data_file(file)
for i, v in enumerate(vels):
    V = v[~np.isnan(v)]
    r = R[~np.isnan(v)]
    fit_vel(r, V)

out = os.path.expanduser("~/thesis/img_ts/"+ratio+"_"+inizio+"_"+str(i*dt)+"_vel.jpg")
plot_R_data_sets(R, vels, "v", dt, out)