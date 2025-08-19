import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, axs = plt.subplots(1, 2, figsize=(13, 6))

for ax in axs:
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off axes, ticks, and grid
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

# Spiral parameters
a = 0.2                     # scale factor
psi_deg = [5, 20]                # pitch angle in degrees

for i, psi in enumerate(psi_deg):
    b= np.tan(np.radians(psi))  # spiral growth factor
    theta = np.linspace(-80//psi * np.pi,  80//psi * np.pi, 40000//psi)  # spiral domain (extended inward and outward)

    # Draw 4 spiral arms
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.radians(angle_deg)
        r = a * np.exp(b * theta)
        x = r * np.cos(theta + angle_rad)
        y = r * np.sin(theta + angle_rad)
        axs[i].plot(x, y, 'k', linewidth=2)

plt.savefig("spirali.pdf", dpi=300)
plt.show()
