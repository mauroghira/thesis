import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from f_gen import *
from f_read import *
from f_plots import *

if len(sys.argv) != 3:
    print("Usage: python plotting.py ratio dt")
    sys.exit(1)

ratio = sys.argv[1]
dt = int(sys.argv[2])
base = "~/thesis/Spiral_pattern/"+ratio

images = []
for i in range(0,11,dt):

    input_file = os.path.expanduser(base+"/img_"+ratio+"_"+str(i)+".pix")
    
    if os.path.exists(input_file):
        image = np.loadtxt(input_file, dtype=float)
        images.append(image)
    else:
        print(f"File {input_file} not found, skipping.")

label = "log column density [g/Cm⁻²]"
pixel_size = 320/544  # AU
shape = 544
extent = [-(shape-1) * pixel_size/2, (shape-1) * pixel_size/2, -(shape-1) * pixel_size/2, (shape-1) * pixel_size/2]

#file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/"+ratio+"_0_5_10_phi.txt")
file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/"+"top_fit_phi.txt")
R, phis = read_R_data_file(file)

for i, phi in enumerate(phis):
    x, y = rphi_to_xy(R, phi, 0)
    out = os.path.expanduser("~/thesis/img_ts/d_fit_"+ratio+"_"+str(i*dt)+".jpg")
    spiral_plot(x,y,images[i], label, extent, out)

#file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/"+ratio+"_0_5_10_vel.txt")
file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/results/top_fit_vel.txt")

R, vels = read_R_data_file(file)

out = os.path.expanduser("~/thesis/img_ts/v_fit_"+ratio+".jpg")
plot_R_data_sets(R, vels, "v", dt, out)