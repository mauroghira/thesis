import numpy as np
from f_read import *
from f_plots import *

def make_r(shape, center=None):
    ny, nx = shape
    y, x = np.indices((ny, nx))
    # true geometric center: (nx-1)/2, (ny-1)/2
    cx = (nx - 1) / 2 if center is None else center[0]
    cy = (ny - 1) / 2 if center is None else center[1]
    return np.hypot(x - cx, y - cy)

def radial_average(image, r, nbins=200, r_max=None):
    nbins = int(nbins)
    if r_max is None:
        r_max = r.max()
    # keep only pixels within r_max
    in_circle = r <= r_max
    r_sel = r[in_circle].ravel()
    img_sel = image[in_circle].ravel()

    bins = np.linspace(0, r_max, nbins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # bin means (vectorized)
    inds = np.digitize(r_sel, bins) - 1
    valid = (inds >= 0) & (inds < nbins)
    inds = inds[valid]
    img_sel = img_sel[valid]

    sums = np.bincount(inds, weights=img_sel, minlength=nbins).astype(float)
    counts = np.bincount(inds, minlength=nbins).astype(float)
    with np.errstate(invalid="ignore"):
        radial_mean = sums / counts
    radial_mean[counts == 0] = np.nan
    return bin_centers, radial_mean


outfile, image, label, pixel_size = read(sys.argv)

# --- usage ---
# 1. Find the minimum non-zero value
nonzero_min = image[image > 0].min()
# 2. Replace zeros with that value
image_fixed = np.where(image == 0, nonzero_min, image)
log_img = np.log10(image_fixed)                           # (H, W)

r = make_r(log_img.shape)                # radii with half-pixel center
r_phys = r * pixel_size               # convert to AU

r_cut = 110.0                            # AU (your chosen physical max)
nbins = log_img.shape[0]                 # or any integer you like

# 1) compute profile only inside r_cut
centers, prof = radial_average(log_img, r_phys, nbins=nbins, r_max=r_cut)

# 2) reconstruct strictly inside r_cut; fill outside with NaN (or 0)
#    IMPORTANT: stop interp from extrapolating with left/right=np.nan
recon_vals = np.interp(r_phys.ravel(), centers, prof, left=np.nan, right=np.nan)
radial_img = recon_vals.reshape(log_img.shape)

# 3) optional hard mask outside r_cut
mask = r_phys <= r_cut
radial_img[~mask] = np.nan               # or 0 if you prefer

# 4) subtraction only where valid
residual = np.where(mask, log_img - radial_img, np.nan)  # or keep original outside

plot_image(residual, pixel_size, label, path=outfile)
plt.show()