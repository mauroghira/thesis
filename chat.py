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
