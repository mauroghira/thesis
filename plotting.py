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

images = []
for i in range(20,31,dt):

    input_file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/m"+ratio)
    #to be modified 
    if ratio == "01":
        i = i%20
        if i == 10:
            input_file += "beta10_000"+str(i)+"_logcolumndensitygcm2_proj.pix"
        else:
            input_file += "beta10_0000"+str(i)+"_logcolumndensitygcm2_proj.pix"
    else:
        input_file += "_000"+str(i)+"_logcolumndensitygcm2_proj.pix"
    
    if os.path.exists(input_file):
        image = np.loadtxt(input_file, dtype=float)
        images.append(image)
    else:
        print(f"File {input_file} not found, skipping.")

label = "log column density [g/Cm⁻²]"
pixel_size = 320/544  # AU
shape = 544
extent = [-(shape-1) * pixel_size/2, (shape-1) * pixel_size/2, -(shape-1) * pixel_size/2, (shape-1) * pixel_size/2]

#file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/"+ratio+"_0_5_10_phi.txt")
file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/"+"topphi.txt")
R, phis = read_R_data_file(file)

for i, phi in enumerate(phis):
    x, y = rphi_to_xy(R, phi, 0)
    
    fig = plt.figure(figsize=(10, 10))
    #plt.scatter(x, y, color="lime", s=10, edgecolor="k", label="Single spial arm")
    plt.plot(x, y, '-', color="lime", label="Single spial arm")
    plt.imshow(images[i], cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(label=label)
    plt.title("Spiral Pattern", size=titlefontsize)
    plt.xlabel("x (AU)", size=labelfontsize)
    plt.ylabel("Y (AU)", size=labelfontsize)
    plt.tick_params(labelsize=tickfontsize)

    plt.legend()
    #angular_grid()
    plt.show()

#file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/"+ratio+"_0_5_10_vel.txt")
file = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/"+"topvel.txt")

R, vels = read_R_data_file(file)

plot_R_data_sets(R, vels, "v", dt)