import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_track_smooot import *

if len(sys.argv) != 2:
    print("Usage: python auto_track.py mass_ratio")
    sys.exit(1)

ratio = sys.argv[1]

images = []
for i in range(20,31):

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

outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/sim_ana/")
label = "log column density [g/Cm⁻²]"
pixel_size = 320/544  # AU

val = float(input("threshold> "))

rm = float(input("R min> "))
rM = float(input("R max> "))
phim = float(input("phi min> "))
phiM = float(input("phi max> "))
name = input("top/bot> ")

for i, image in enumerate(images):
    print("year ", i)
    sk = input("wanna skip?> ")
    if sk == "y":
        continue

    peaks = find_2d_peaks(image, val)
    rows = np.array([peak[0] for peak in peaks])
    cols = np.array([peak[1] for peak in peaks])
    spiral = np.zeros_like(image)
    spiral[rows, cols] = image[rows, cols]

    rep = "y"
    while rep== "y":
        xy_neighbors = filter_peaks_by_rphi(peaks, image.shape[0], pixel_size, rm, rM, phim, phiM)
        plot_neighbors(xy_neighbors, image, pixel_size, label)
        plt.show()

        with open("operations.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        param_iter = iter(lines)
        while True:
            try:
                what = next(param_iter)
                if what == "stop":
                    break
                pm = float(next(param_iter))
                pM = float(next(param_iter))
                xy_neighbors = modify_r_by_phi_extremes(xy_neighbors, image.shape[0], pm, pM, what)
            
            except StopIteration:
                print("Reached end of input file.")
                break
        
        plot_neighbors(xy_neighbors, image, pixel_size, label)
        plt.show()
            
        while True:
            what = input("in/out/avg/del/stop> ")
            if what == "stop":
                break

            pm = float(input("phi min> "))
            pM = float(input("phi max> "))
            xy_neighbors = modify_r_by_phi_extremes(xy_neighbors, image.shape[0], pm, pM, what)

            plot_neighbors(xy_neighbors, image, pixel_size, label)
            plt.show()

        rep = input("try again?> ")
    
    sv = input("wanna save?> ")
    if sv != "y":
        continue
    file = outfile+str(i)+"_"+name+".txt"
    save_rphi(xy_neighbors, file, phiM, pixel_size, image.shape[0])