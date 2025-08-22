from astropy.io import fits #library to analyse fits files
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#for the fits files simply give the path massratio/inclination/year/
if len(sys.argv)==2 and "inc" in sys.argv[1]:
    # If the input is a FITS file, read it
    folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
    file="data_1300/RT.fits.gz"
    name = folder+file
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"flux_7.jpg")

    hdul = fits.open(name)
    image_data = hdul[0].data
    hdul.close()

    image = image_data[0, 0, 0, :, :]  # select first frame
    label = "Flux [W/(m⁻² pixel⁻¹)]"
    pixel_size = 300/image.shape[0] # AU
    hh = 0.2e-19
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    plt.imshow(image, cmap="inferno", extent=extent, origin="lower")
    plt.colorbar(label="Flux [W/(m⁻² pixel⁻¹)]")
    plt.title("Total flux")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

#for the hydrodynamical simulations give the path massratio filename
elif len(sys.argv)==3:
    # If the input is a text file, read it
    outfile = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/sim_ana/y5.jpg")
    path = os.path.expanduser("~/thesis/Spiral_pattern/"+sys.argv[1]+"/"+sys.argv[2])
    image = np.loadtxt(path, dtype=float)
    label = "log column density [g/Cm⁻²]"
    pixel_size = 320/image.shape[0]  # AU
    hh = 3.5
    extent = [-image.shape[1] * pixel_size/2, image.shape[1] * pixel_size/2, -image.shape[1] * pixel_size/2, image.shape[0] * pixel_size/2]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Disc density")

    images = [image, 10**image]

    for i, axs in enumerate(ax):
        pcm = axs.imshow(images[i], cmap="inferno", extent=extent, origin="lower")
        fig.colorbar(pcm, ax=axs, label="(log) column density [g/Cm⁻²]")
        axs.set_xlabel("x (AU)")
        axs.set_ylabel("Y (AU)")

    plt.tight_layout()
    #plt.savefig(outfile, bbox_inches="tight")
    plt.show()

else:
    print("Invalid input. Please provide a valid FITS file or simulation data file.")
    sys.exit(1)

#backup no good
def vicini(points, size, bound=np.inf, start="b", neig=10):
    #work on points in the R phi representation to avoid infinite loops
    tree = KDTree(points)
    if start == "b":
        #start from the bottom
        partial = points[points[:, 1] < 0]
    elif start == "t":
        #start from the top
        partial = points[points[:, 1] > 0]
    else:
        print("Invalid start point. Use 'b' for bottom or 't' for top.")
        return None

    max = np.argmax(partial[:, 0])
    mean = np.mean(points[:, 0])
    query_point = points[max]  # pick one point
    qp = np.array([])
    #print(points[max], mean)

    all_ind = []
    # track the spiral by looping the neighbour search
    while query_point[0] > mean:
        #qp = np.append(qp, query_point)
        distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)

        temp_bound=bound
        #""" use this code placinb the bound to infinity in the query
        
        mask = distances <= bound
        tem_distances = distances[mask]
        while tem_distances.size == 0:
            print("No points inside found, increasing bound")
            temp_bound += 0.1
            distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)
            mask = distances <= temp_bound
            tem_distances = distances[mask]
        
        distances = distances[mask]
        indices   = indices[mask]
        ind_dmax = np.argmax(distances)

        """#use this with the bound+bound in the query above

        while np.any(np.isinf(distances)):
            print("Infinite distance found, increasing bound")
            temp_bound += 0.1
            distances, indices = tree.query(query_point, k=neig, distance_upper_bound=temp_bound)
        ind_dmax = np.argmax(distances)

        #"""

        # i assume the spiral to be monotonous in the R direction
        # so i can folow the arm by decreasing the radius
        Rmax, phimax = points[indices[ind_dmax]]
        while Rmax >= query_point[0]:
            distances = np.delete(distances, ind_dmax)
            indices = np.delete(indices, ind_dmax)
            if indices.size == 0:
                print("no more monotonous - quitting")
                break
            ind_dmax = np.argmax(distances)
            Rmax, phimax = points[indices[ind_dmax]]
        
        """
        while ((phimax-query_point[1] + np.pi) % (2*np.pi) - np.pi) < 0:
            disc += 1
            #print("not counterclockwise - deleting")
            distances = np.delete(distances, ind_dmax)
            indices = np.delete(indices, ind_dmax)
            if indices.size == 0:
                print("no longer counterclockwise - quitting")
                break
            ind_dmax = np.argmax(distances)
            Rmax, phimax = points[indices[ind_dmax]]
        #"""

        """
        if phimax*query_point[1] < 0 and (phimax<-np.pi/2 or query_point[1]<np.pi/2):
            while phimax <= query_point[1]-2*np.pi:
                distances = np.delete(distances, ind_dmax)
                indices = np.delete(indices, ind_dmax)
                if indices.size == 0:
                    print("no more monotonous - quitting")
                    break
                ind_dmax = np.argmax(distances)
                Rmax, phimax = points[indices[ind_dmax]]
            if indices.size == 0:
                print("no more monotonous - quitting")
                break

        elif start == "t" and query_point[1] < 0:
            while phimax <= query_point[1]+2*np.pi:
                distances = np.delete(distances, ind_dmax)
                indices = np.delete(indices, ind_dmax)
                if indices.size == 0:
                    print("no more monotonous - quitting")
                    break
                ind_dmax = np.argmax(distances)
                Rmax, phimax = points[indices[ind_dmax]]
            if indices.size == 0:
                print("no more monotonous - quitting")
                break
        """
        
        if indices.size == 0:
            print("no more monotonous - quitting")
            break
        query_point = points[indices[ind_dmax]]  # pick the farthest point
        #print("Query point:", query_point, "Distance:", distances[ind_dmax])

        all_ind.extend(indices.tolist())  # add all elements, not the array itself

    #remove duplicates
    all_ind = np.unique(all_ind)
    vic = points[all_ind]
    return vic

def vicini(points, bound=np.inf, start="b", neig=10):
    #work on points in the R phi representation to avoid infinite loops
    tree = KDTree(points)
    if start == "b":
        #start from the bottom
        partial = points[points[:, 1] < 0]
    elif start == "t":
        #start from the top
        partial = points[points[:, 1] > 0]
    else:
        print("Invalid start point. Use 'b' for bottom or 't' for top.")
        return None

    max = np.argmax(partial[:, 0])
    mean = np.mean(points[:, 0])
    query_point = partial[max]  # pick one point

    listed = set() #for easier lookup later

    # track the spiral by looping the neighbour search
    while query_point[0] > mean:
        #qp = np.append(qp, query_point)
        distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)

        mask = (distances > 0)
        distances = distances[mask]
        indices   = indices[mask]

        out=0
        # mask within adaptive bound
        temp_bound = bound
        mask = (distances <= temp_bound)
        while not np.any(mask):
            print("boh")
            out += 1
            temp_bound += 0.1
            mask = (distances <= temp_bound)

        distances = distances[mask]
        indices   = indices[mask]

        # remove already listed points
        mask_new = [i not in listed for i in indices]
        distances = distances[mask_new]
        indices   = indices[mask_new]

        if indices.size == 0:
            print("no new neighbors matching conditons - stopping")
            break

        listed.update(indices.tolist())   # add multiple at once
        ind_dmax = np.argmax(distances)
        query_point = points[indices[ind_dmax]]  # pick the farthest point
        print("Query point:", query_point, "Distance:", distances[ind_dmax], "out ", out)

    print(len(list(listed)))
    #remove duplicates
    vic = points[list(listed)] #transform set into list to use as indices
    return vic


def vic(points, bound=2, max_b=5, max_r=4.5, start="b"):
    tree = KDTree(points)

    # choose starting points
    if start == "b":
        partial = points[points[:, 1] < 0]
    elif start == "t":
        partial = points[points[:, 1] > 0]
    else:
        print("Invalid start point. Use 'b' for bottom or 't' for top.")
        return None

    # pick the bottommost/topmost point with largest radius
    max_idx = np.argmax(partial[:, 0])
    mean_r  = np.mean(points[:, 0])
    query_point = partial[max_idx]

    listed = set()

    # walk until radius is below mean
    while query_point[0] > mean_r:
        temp_bound = bound
        out = 0

        # expand radius until we find neighbors
        indices = []
        while not indices:
            indices = tree.query_ball_point(query_point, r=temp_bound)
            # drop already visited points
            indices = [i for i in indices if i not in listed and not np.all(points[i] == query_point)]
            
            #delete point too far in radial distance
            r_current = query_point[0]
            indices = [i for i in indices if abs(points[i][0] - r_current) <= max_r]
            
            if not indices:
                out += 1
                temp_bound += 0.1
                if temp_bound > max_b:  # avoid infinite loop
                    print("giving up, bound expanded too much")
                    break

        if not indices:
            print("no new neighbors matching conditions - stopping")
            break

        # pick farthest neighbor
        distances = np.linalg.norm(points[indices] - query_point, axis=1)
        ind_dmax = np.argmax(distances)

        # update state
        listed.update(indices)   # add multiple at once
        query_point = points[indices[ind_dmax]]  # pick the farthest point
        
        print("next Query point:", query_point, "Distance:", distances[ind_dmax], "out", out)

    print("Total collected:", len(listed))
    return points[list(listed)]