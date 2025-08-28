import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import find_peaks 
from fuctions import *

outfile, image, label, pixel_size = read(sys.argv)

#parse
while True:
    print("==============================")
    print("Insert: string value - Available commands:")
    print("______")
    print("    find h   //find the points above the threshold h")
    print("    spot     //isolate a spiral arm")
    print("    track    //track the points")
    print("    save n   //save R-phi data of the spiral arm at year n")
    print("")
    print("    plot 0   // plot the original image")
    print("    plot 1   // plot the spiral arms found")
    print("    plot 2   // plot the single spiral arm found ")
    print("    plot 3   // plot the complete Rphi map")
    print("    plot 4   // plot the single spiral arm in Rphi map")
    print("")
    print("    quit n   // exit the program")
    print("______")

    #read command
    cmd = input("string> ")

    if cmd == "quit":
        exit()
	
    elif cmd == "find":
        val = float(input("threshold> "))
        peaks = find_2d_peaks(image, val)
        rows = np.array([peak[0] for peak in peaks])
        cols = np.array([peak[1] for peak in peaks])
        spiral = np.zeros_like(image)
        spiral[rows, cols] = image[rows, cols]

    elif cmd == "spot":
        try:
            peaks
        except NameError:
            print("No spiral arms found. Please run 'find h' first.")
            continue

        rm = float(input("R min> "))
        rM = float(input("R max> "))
        phim = float(input("phi min> "))
        phiM = float(input("phi max> "))
        lim = float(input("lim> "))
        xy_neighbors = filter_peaks_by_rphi(peaks, image.shape[0], pixel_size, rm, rM, phim, phiM, lim)

    elif cmd == "track":
        try:
            peaks
        except NameError:
            print("No spiral arms found. Please run 'find h' first.")
            continue
        
        val = float(input("radius> "))
        max = float(input("maximum radial distance accepted> "))
        print("Insert t to start tracking from the top, b from the bottom")
        st_point = input("start> ")
        rphi_neighbors = neig(peaks, size=image.shape[0], bound=val, max_dr=max, start=st_point)
        if rphi_neighbors is not None:
            xn, yn = rphi_to_xy(rphi_neighbors, image.shape[0])
            xy_neighbors = np.column_stack((xn, yn))		
	
    elif cmd=="save":
        try:
            xy_neighbors
        except NameError:
            print("No spiral arms found. Please run 'track d' first.")
            continue
        val = int(input("year> "))
        name = input("top/bot> ")
        file = outfile+str(val)+"_"+name+".txt"
        save_rphi(xy_neighbors, file, phiM, pixel_size, image.shape[0])

    elif cmd == "plot":
        val = int(input("number> "))
        match val:
            case 0:
                plot_image(image, pixel_size, label, path=outfile)
            
            case 1:
                try:
                    spiral
                except NameError:
                    print("No spiral arms found. Please run 'find h' first.")
                    continue
                plot_image(spiral, pixel_size, label, path=outfile)
            
            case 2:
                try:
                    xy_neighbors
                except NameError:
                    print("No spiral arms found. Please run 'track d' first.")
                    continue
                plot_neighbors(xy_neighbors, image, pixel_size, label, path=outfile)
            
            case 3:
                try:
                    peaks
                except NameError:
                    print("No spiral arms found. Please run 'find h' first.")
                    continue
                r, phi = xy_to_rphi(peaks[:, 0], peaks[:,1], image.shape[0])
                plot_rphi_map(r, phi, pixel_size)
            
            case 4:
                try:
                    xy_neighbors
                except NameError:
                    print("No spiral arms found. Please run 'track d' first.")
                    continue
                r = rphi_neighbors[:, 0]
                phi = rphi_neighbors[:, 1]
                plot_rphi_map(r, phi, pixel_size)
            
            case _:
                print("Invalid plot command. Please use 0, 1, 2, 3, or 4.")

        save=input("save image? y or n> ")
        if save=="y":
            name = input("filename> ")
            plt.savefig(outfile+name, bbox_inches="tight")
        plt.show()

    else:
        print("Invalid command. Please try again.")