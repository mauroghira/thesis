import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Example dataset: random scatter + a "spiral arm"
np.random.seed(0)
random_points = np.random.rand(100, 2) * 10   # background noise
theta = np.linspace(0, 4*np.pi, 50)
spiral_x = theta * np.cos(theta)
spiral_y = theta * np.sin(theta)
spiral_points = np.column_stack([spiral_x + 5, spiral_y + 5])  # shift to center

# Combine into one dataset
points = np.vstack([random_points, spiral_points])

# Build KDTree
tree = KDTree(points)

# --- Example 1: Find nearest neighbors of one point
query_point = spiral_points[40]     # pick one spiral point
distances, indices = tree.query(query_point, k=10)  # 10 nearest neighbors
vicini = points[indices]
print("Nearest neighbors:", points[indices])

# --- Example 2: Extract points within a radius
radius = 1.0
indices_in_radius = tree.query_ball_point(query_point, r=radius)
neighbors = points[indices_in_radius]

# Plot everything
plt.figure(figsize=(10,10))
plt.scatter(vicini[:,0], vicini[:,1], color="lime", s=60, edgecolor="k", label="Nearest neighbors")
plt.scatter(points[:,0], points[:,1], s=20, alpha=0.5, label="All points")
plt.scatter(spiral_points[:,0], spiral_points[:,1], color="red", label="Spiral")
#plt.scatter(neighbors[:,0], neighbors[:,1], color="lime", s=60, edgecolor="k", label="Neighbors")
plt.scatter(*query_point, color="blue", s=100, marker="x", label="Query point")
plt.legend()
plt.title("KDTree neighbor search")
plt.show()

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

    visited = set()
    visited.add(tuple(query_point))  # store as tuple for hashing

    all_ind = []

    while query_point[0] > mean:
        distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)

        # mask within adaptive bound
        temp_bound = bound
        mask = distances <= temp_bound
        while not np.any(mask):
            temp_bound += 0.1
            distances, indices = tree.query(query_point, k=neig, distance_upper_bound=np.inf)
            mask = distances <= temp_bound

        distances = distances[mask]
        indices   = indices[mask]

        # remove already visited points
        mask_new = [tuple(points[i]) not in visited for i in indices]
        distances = distances[mask_new]
        indices   = indices[mask_new]

        if indices.size == 0:
            print("no new neighbors - stopping")
            break

        # keep only CCW neighbors
        dphi = (points[indices,1] - query_point[1] + np.pi) % (2*np.pi) - np.pi
        mask_ccw = dphi > 0
        distances = distances[mask_ccw]
        indices   = indices[mask_ccw]

        if indices.size == 0:
            print("no counterclockwise neighbors - stopping")
            break

        # pick farthest CCW
        ind_dmax = np.argmax(distances)
        query_point = points[indices[ind_dmax]]
        visited.add(tuple(query_point))

        all_ind.append(indices[ind_dmax])

        #print("Next query point:", query_point)
    #remove duplicates
    all_ind = np.unique(all_ind)
    vic = points[all_ind]
    return vic