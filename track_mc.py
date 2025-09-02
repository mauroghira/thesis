import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.io import fits #library to analyse fits files

from f_gen import *
from f_read import *
from f_plots import *
from f_save import *
from f_track import *

if len(sys.argv) != 9:
    print("Usage: python track_mc.py mass_ratio threshold max Rmin Rmax phimin phimax arm")
    sys.exit(1)

ratio = sys.argv[1]
val = float(sys.argv[2])
mx = float(sys.argv[3])
rm = float(sys.argv[4])
rM = float(sys.argv[5])
phim = float(sys.argv[6])
phiM = float(sys.argv[7])
name = sys.argv[8]

images = []
for i in range(0,11):

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

outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/obs_data/")
label = "log flux [W/(m² px)]"
pixel_size = 320/512  # AU

ex = 1e-20

for i, image in enumerate(images):
    print("year ", int(i*10/(len(images)-1)))
    sk = input("wanna skip?> ")
    if sk == "y":
        continue

    peaks = find_2d_peaks(image, val*ex, mx*ex)
    rows = np.array([peak[0] for peak in peaks])
    cols = np.array([peak[1] for peak in peaks])
    spiral = np.zeros_like(image)
    spiral[rows, cols] = image[rows, cols]

    rep = "y"
    while rep== "y":
        xy_neighbors = filter_peaks_by_rphi(peaks, image.shape[0], pixel_size, rm, rM, phim, phiM)
        plot_neighbors(xy_neighbors, images[i], pixel_size, label, -1e-21)
        plt.show()

        opera = os.path.expanduser("~/thesis/Spiral_pattern/"+ratio+"/operations/"+str(int(i*10/(len(images)-1)))+"_mc_"+name+".txt")

        with open(opera, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                parts = line.split()
                
                what = parts[0]
                if what == "stop":
                    break

                pm = float(parts[1])
                pM = float(parts[2])
                xy_neighbors = modify_r_by_phi_extremes(xy_neighbors, image.shape[0], pm, pM, what)
        
        plot_neighbors(xy_neighbors, images[i], pixel_size, label, -1e-21)
        plt.show()
            
        while True:
            what = input("in/out/avg/fit/del/stop> ")
            if what == "stop":
                break

            pm = float(input("phi min> "))
            pM = float(input("phi max> "))
            xy_neighbors = modify_r_by_phi_extremes(xy_neighbors, image.shape[0], pm, pM, what)

            plot_neighbors(xy_neighbors, images[i], pixel_size, label, -1e-21)
            plt.show()

        rep = input("try again?> ")
    
    sv = input("wanna save?> ")
    if sv != "y":
        continue
    file = outfile+str(int(i*10/(len(images)-1)))+"_"+name+".txt"
    save_rphi(xy_neighbors, file, phiM, pixel_size, image.shape[0])