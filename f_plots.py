import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from f_gen import kepler

titlefontsize, labelfontsize, tickfontsize=20, 16, 14
markersize, linewidth = 6, 1

#############
#===========================================================
############# function to plot the original image or the peaks

def plot_image(image, pixel_size, label, vmin=None):
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if vmin is None:
        im = ax.imshow(image, cmap="inferno", origin="lower", extent=extent)
    else:
        im = ax.imshow(image, cmap="inferno", vmin=vmin, origin="lower", extent=extent)

    # Get position of the main axis
    pos = ax.get_position()
    # Create new axis on top of the image, same width
    cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=labelfontsize, labelpad=15)
    cbar.ax.xaxis.set_ticks_position("top") 
    cbar.ax.xaxis.set_label_position("top")

    #ax.set_title("Spiral Pattern", size=titlefontsize)
    ax.set_xlabel("x (AU)", size=labelfontsize)
    ax.set_ylabel("Y (AU)", size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)

    #angular_grid(ax)
    #plt.show()


#############
#===========================================================
############# function to plot the spiral arm neighbors

def plot_neighbors(xy_neighbors, image, pixel_size, label, vmin=None):
    scaled_neighbors = (xy_neighbors-(image.shape[0]-1)/2) * pixel_size
    extent = [-(image.shape[1]-1) * pixel_size/2, (image.shape[1]-1) * pixel_size/2, -(image.shape[0]-1) * pixel_size/2, (image.shape[0]-1) * pixel_size/2]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if vmin is None:
        im = ax.imshow(image, cmap="inferno", origin="lower", extent=extent)
    else:
        im = ax.imshow(image, cmap="inferno", vmin=vmin, origin="lower", extent=extent)

    # Get position of the main axis
    pos  = ax.get_position()
    # Create new axis on top of the image, same width
    cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=labelfontsize, labelpad=15)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=tickfontsize)

    #ax.set_title("Spiral Pattern", size=titlefontsize)
    ax.set_xlabel("x (AU)", size=labelfontsize)
    ax.set_ylabel("Y (AU)", size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)
    ax.scatter(scaled_neighbors[:, 1], scaled_neighbors[:, 0], color="lime", s=1, label="Single spial arm")
    #ax.plot(scaled_neighbors[:, 1], scaled_neighbors[:, 0], '-', color="lime", label="Single spial arm")

    angular_grid(ax)
    
    ax.legend()

    #plt.show()

#############
#===========================================================
############# function to plot angular grid to easy trackk
def angular_grid(ax):
    # Angular grid (spokes)
    angles = np.deg2rad(np.arange(0, 360, 20))  # every 30 degrees

    # Radial grid (concentric circles)
    radii = np.linspace(20, 100, 5)
    circle = np.linspace(0, 2*np.pi, 300)
    for r in radii:
        ax.plot(r*np.cos(circle), r*np.sin(circle), color='lime', linestyle='--', alpha=0.5)

    # Radii (spokes at given angles)
    r_max = radii.max()
    for angle in angles:
        ax.plot([0, r_max*np.cos(angle)], [0, r_max*np.sin(angle)], color='lime', linestyle='--', alpha=0.5)

    plt.gca().set_aspect('equal')


#############
#===========================================================
############# function to plot a single Rphi map of the peaks
def plot_rphi_map(r, phi, peaks_size):
    r = r * peaks_size  # scale radius
    plt.figure(figsize=(10, 10))
    plt.scatter(r, phi, c='black', s=5, label="Spiral Arms")
    plt.title("R-$\phi$ Map of Peaks")
    plt.xlabel("Radius (AU)", size=labelfontsize)
    plt.ylabel("$\phi$ (radians)", size=labelfontsize)
    plt.tick_params(labelsize=tickfontsize)

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
    plt.xlabel('R (AU)', size=labelfontsize)
    plt.ylabel('$\phi$ (radians)', size=labelfontsize)
    plt.tick_params(labelsize=tickfontsize)
    plt.title(title, size=titlefontsize)

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
    plt.ylabel('R (AU)', size=labelfontsize)
    plt.xlabel('$\phi$ (radians)', size=labelfontsize)
    plt.tick_params(labelsize=tickfontsize)
    plt.title(title, size=titlefontsize)

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
    radii, omegaK_year = kepler(int_dt[0][:,0])

    plt.figure(figsize=(9, 9))
    for i, data in enumerate(int_dt):
        if i*dt != 10:
            label=fr'Year {i*dt} $\rightarrow$ {dt*(i+1)}'
        else:
            label=fr'Year 0 $\rightarrow$ 10'
        plt.plot(data[:, 0], data[:, 1], label=label)
    plt.plot(radii, omegaK_year, '.', label="Keplerian angular velocity")
    plt.xlabel('R (AU)', size=labelfontsize)
    plt.ylabel(r"$\frac{d\phi}{dt} = \Omega(R)$ (radians/year)", size=labelfontsize)
    plt.title("Angular pattern velocity", size=titlefontsize)
    plt.tick_params(labelsize=tickfontsize)
    plt.legend()
    plt.grid(True)
    plt.show()


#############
#===========================================================
############# function to plot data read from file
def plot_R_data_sets(x, ys, type="p", dt=5, outfile=""):
    plt.figure(figsize=(9, 9))
    for i, y in enumerate(ys):
        if type == "v":
            if i == len(ys)-1:
                label = fr'Year 0 $\rightarrow$ 10'
            else:
                label = fr'Year {i*dt} $\rightarrow$ {dt*(i+1)}'
        else:
            label = f'Year {i*dt}'

        plt.plot(x, y, label=label)
    
    if type=="v":
        radii, omegaK_year = kepler(x)
        plt.plot(radii, omegaK_year, '.', label="Keplerian angular velocity")
        ax_lab = r"$\frac{d\phi}{dt} = \Omega(R)$ (radians/year)"
        title = "Angular patern speeds"
    else:
        ax_lab = '$\phi$ (radians)'
        title = "R-$\phi$ Map of the spiral arm"
            # set y ticks in multiples of π/2
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/4))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: f"{int(round(val/np.pi))}π" if np.isclose(val % np.pi, 0) else f"{val/np.pi:.1f}π"
        ))

    plt.xlabel('R (AU)', size=labelfontsize)
    plt.ylabel(ax_lab, size=labelfontsize)
    plt.title(title, size=titlefontsize)
    plt.tick_params(labelsize=tickfontsize)

    plt.legend()
    plt.grid(True)
    #plt.savefig(outfile, bbox_inches="tight")
    #plt.show()


#############
#===========================================================
############# function to plot spiral and track
def spiral_plot(x,y, image, label, extent, outfile, vmin):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if vmin is None:
        im = ax.imshow(image, cmap="inferno", origin="lower", extent=extent)
    else:
        im = ax.imshow(image, cmap="inferno", vmin=vmin, origin="lower", extent=extent)
    # Get position of the main axis
    pos  = ax.get_position()
    # Create new axis on top of the image, same width
    cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=labelfontsize, labelpad=15)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=tickfontsize)
    
    ax.plot(x, y, '-', color="lime", label="Single spial arm")
    #ax.set_title("Spiral Pattern", size=titlefontsize)
    ax.set_xlabel("x (AU)", size=labelfontsize)
    ax.set_ylabel("Y (AU)", size=labelfontsize)
    ax.tick_params(labelsize=tickfontsize)

    ax.legend()
    #angular_grid()
    #plt.savefig(outfile, bbox_inches="tight")

    plt.show()