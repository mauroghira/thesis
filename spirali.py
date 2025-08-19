import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

ax.axis('off')  # Turn off axes, ticks, and grid
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
#ax.set_xticks(np.arange(-3, 4, 1))
#ax.set_yticks(np.arange(-3, 4, 1))
#ax.grid(False, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

# Spiral parameters
a = 0.2                     # scale factor
psi_deg = 30                # pitch angle in degrees
b = np.tan(np.radians(psi_deg))  # spiral growth factor
theta = np.linspace(-4 * np.pi, 4 * np.pi, 2000)  # spiral domain (extended inward and outward)

# Draw 4 spiral arms
for angle_deg in [0, 90, 180, 270]:
    angle_rad = np.radians(angle_deg)
    r = a * np.exp(b * theta)
    x = r * np.cos(theta + angle_rad)
    y = r * np.sin(theta + angle_rad)
    ax.plot(x, y, 'k', linewidth=2)

# Draw radial direction (R)
ax.plot([1.25,1.25], [-3, 3], 'k--', linewidth=1)
ax.plot([0,3], [0, 0], 'k--', linewidth=1)
#ax.text(2.9, -0.3, 'R', fontsize=14, verticalalignment='center')

# Doppler-shifted phase velocity vector \tilde{v}_p
vp = 1.5
ax.arrow(1.25, 0, vp, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
ax.text(1.7, 0.1, r'$\tilde{v}_p$', fontsize=14, color='blue')

# Projected velocity vector \tilde{v}_p cos \psi
vp_proj = vp * np.cos(np.radians(psi_deg))
vx = vp_proj * np.cos(np.radians(psi_deg))
vy = -vp_proj * np.sin(np.radians(psi_deg))
ax.arrow(1.25, 0, vx, vy, head_width=0.1, head_length=0.1, fc='orange', ec='orange', linewidth=2)
ax.text(vx+0.2, vy, r'$\tilde{v}_p \cos \psi$', fontsize=14, color='orange')

# Dashed line showing projection
ax.plot([1.25+vp, 1.25+vx], [0, vy], 'k--', linewidth=1)

# Pitch angle arc
arc_theta = np.radians(np.linspace(0, psi_deg, 100))
arc_r = 1
arc_y = arc_r * np.cos(arc_theta)
arc_x = 1.25 + arc_r * np.sin(arc_theta)
ax.plot(arc_x, arc_y, color='purple', linewidth=2)
ax.text(1.55, 1.1, r'$\psi$', fontsize=14, color='purple')

# Pitch angle arc
arc_theta = np.radians(np.linspace(0, psi_deg, 100))
arc_r = 1
arc_y = -arc_r * np.sin(arc_theta)
arc_x = 1.25 + arc_r * np.cos(arc_theta)
ax.plot(arc_x, arc_y, color='purple', linewidth=2)
ax.text(2.35, -0.3, r'$\psi$', fontsize=14, color='purple')

ax.plot([1.25-3*np.sin(np.radians(psi_deg)),1.25+3*np.sin(np.radians(psi_deg))], [-3*np.cos(np.radians(psi_deg)), 3*np.cos(np.radians(psi_deg))], 'k--', linewidth=1)

# Create a circle with center at (0,0) and radius 5
circle = plt.Circle((0, 0), 1.25, fill=False, color='red', linewidth=1, linestyle='-.')
# Add the circle to the plot
ax.add_artist(circle)

# Labels and layout
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_title('Logarithmic Spiral with Pitch Angle 30°')
plt.tight_layout()

# Save to file
plt.savefig("shock.pdf", dpi=300)
plt.show()
