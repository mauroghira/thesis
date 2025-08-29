import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from astropy.constants import G, au, M_sun
import astropy.units as u

#############
#===========================================================
############# function to plot the original image or the peaks

def plot_image(image, pixel_size, label, path=""):
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(label=label)
    plt.title("Spiral Pattern")
    plt.xlabel("x (AU)")
    plt.ylabel("Y (AU)")
    #plt.show()


#############
#===========================================================
############# function to plot the spiral arm neighbors

def plot_neighbors(xy_neighbors, image, pixel_size, label, path=""):
    scaled_neighbors = (xy_neighbors-(image.shape[0]-1)/2) * pixel_size
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(scaled_neighbors[:, 1], scaled_neighbors[:, 0], color="lime", s=10, edgecolor="k", label="Single spial arm")
    #plt.plot(scaled_neighbors[:, 1], scaled_neighbors[:, 0], '-', color="lime", label="Single spial arm")
    plt.imshow(image, cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(label=label)
    plt.title("Spiral Pattern")
    plt.xlabel("x (AU)")
    plt.ylabel("Y (AU)")
    plt.legend()

    angular_grid()

    #plt.show()

#############
#===========================================================
############# function to plot angular grid to easy trackk
def angular_grid():
    # Angular grid (spokes)
    angles = np.deg2rad(np.arange(0, 360, 20))  # every 30 degrees

    # Radial grid (concentric circles)
    radii = np.linspace(20, 100, 5)
    circle = np.linspace(0, 2*np.pi, 300)
    for r in radii:
        plt.plot(r*np.cos(circle), r*np.sin(circle), color='black', linestyle='--', alpha=0.3)

    # Radii (spokes at given angles)
    r_max = radii.max()
    for angle in angles:
        plt.plot([0, r_max*np.cos(angle)], [0, r_max*np.sin(angle)], color='black', linestyle='--', alpha=0.3)


    plt.gca().set_aspect('equal')


#############
#===========================================================
############# function to plot a single Rphi map of the peaks
def plot_rphi_map(r, phi, peaks_size):
    r = r * peaks_size  # scale radius
    plt.figure(figsize=(10, 10))
    plt.scatter(r, phi, c='black', s=5, label="Spiral Arms")
    plt.title("R-$\phi$ Map of Peaks")
    plt.xlabel("Radius (AU)")
    plt.ylabel("$\phi$ (radians)")

    # set y ticks in multiples of π/2
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid()
    #plt.show()


#############
#===========================================================
############# function to plot all the phi(R) maps
def plot_all_phi_r(int_dt, title, dt):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 0], data[:, 1], label=f'Year {dt*i}')
    plt.xlabel('R (AU)')
    plt.ylabel('$\phi$ (radians)')
    plt.title(title)

    # set y ticks in multiples of π/2
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid(True)
    plt.show()

#############
#===========================================================
############# function to plot all the R(phi) maps
def plot_all_r_phi(int_dt, title, dt):
    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        plt.plot(data[:, 1], data[:, 0], label=f'Year {i*dt}')
    plt.ylabel('R (AU)')
    plt.xlabel('$\phi$ (radians)')
    plt.title(title)

    # set y ticks in multiples of π/2
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
    ))

    plt.legend()
    plt.grid(True)
    plt.show()


#############
#===========================================================
############# function to plot the velocities
def plot_vel(int_dt, dt):
    #convert units for consistency
    radii = np.linspace(np.min(int_dt[0][:,0]), np.max(int_dt[0][:,0]), 100) * u.au
    omegaK = np.sqrt(G*M_sun/radii**3)
    # Convert to rad/year
    omegaK_year = omegaK.to(1/u.yr)   # "1/u.yr" means frequency in cycles per year (rad/year)
    #print(omegaK_year)

    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        if i*dt != 10:
            label=fr'Year {i*dt} $\rightarrow$ {dt*(i+1)}'
        else:
            label=fr'Year 0 $\rightarrow$ 10'
        plt.plot(data[:, 0], data[:, 1], label=label)
    plt.plot(radii, omegaK_year, '.', label="Keplerian angular velocity")
    plt.xlabel('R (AU)')
    plt.ylabel(r"$\frac{d\phi}{dt} = \Omega(R)$ (radians/year)")
    plt.title("Angular pattern velocity")

    plt.legend()
    plt.grid(True)
    plt.show()