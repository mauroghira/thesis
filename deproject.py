import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import sys
from astropy.io import fits

# --- Inputs ---
# image: 2D numpy array (grayscale)
# Provide center if known; else defaults to image center
def fit_spiral_with_deprojection(img, inclination_deg=15.0, center=None, PA_deg=None, threshold=None):
    #img = np.asarray(image)
    ny, nx = img.shape

    # 1) Choose/estimate center
    if center is None:
        x0, y0 = nx/2.0, ny/2.0
    else:
        x0, y0 = center

    # 2) Pick pixels belonging to the spiral (simple threshold or edges)
    if threshold is None:
        threshold = img.mean() + img.std()
    ys, xs = np.where(img > threshold)

    if len(xs) < 50:
        # Fall back to edges if thresholding is too sparse
        im8 = np.clip((img - img.min()) / (img.ptp() + 1e-9) * 255, 0, 255).astype(np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(im8, (5,5), 0), 50, 150)
        ys, xs = np.where(edges > 0)

    # Shift to center
    X = xs - x0
    Y = ys - y0

    # 3) Estimate PA (if not given) from 2D covariance of bright pixels
    #    PA = angle of major axis (in radians), measured from +x toward +y
    if PA_deg is None:
        pts = np.vstack([X, Y])  # shape (2, N)
        C = np.cov(pts)          # 2x2 covariance
        evals, evecs = np.linalg.eigh(C)
        major = evecs[:, np.argmax(evals)]
        PA = np.arctan2(major[1], major[0])
        PA_deg_est = np.degrees(PA)
    else:
        PA = np.radians(PA_deg)
        PA_deg_est = PA_deg

    # 4) Rotate points by -PA so major axis is horizontal
    c, s = np.cos(-PA), np.sin(-PA)
    Xr = c*X - s*Y
    Yr = s*X + c*Y

    # 5) Deproject: stretch the (projected) minor axis by 1/cos(i)
    i = np.radians(inclination_deg)
    deproj_factor = 1.0 / np.cos(i)  # ~1.0353 for 15°
    Yd = Yr * deproj_factor
    Xd = Xr

    # 6) Convert to polar in the deprojected (face-on) plane
    r = np.hypot(Xd, Yd)
    theta = np.arctan2(Yd, Xd)
    theta = np.unwrap(theta)

    # Optionally filter out tiny/huge radii (robustness)
    good = (r > np.percentile(r, 5)) & (r < np.percentile(r, 95))
    r_fit = r[good]
    theta_fit = theta[good]

    # 7) Fit logarithmic spiral: r = a * exp(b * theta)
    def log_spiral(theta_vals, a, b):
        return a * np.exp(b * theta_vals)

    # Initial guesses
    p0 = [np.median(r_fit[r_fit > 0]) if np.any(r_fit > 0) else 1.0, 0.1]
    popt, pcov = curve_fit(log_spiral, theta_fit, r_fit, p0=p0, maxfev=20000)
    a_fit, b_fit = popt

    # 8) Pitch angle (face-on)
    # tan(phi) = 1/b  ->  phi = arctan(1/b)
    phi = np.degrees(np.arctan(1.0 / b_fit))

    # 9) Create a smooth fitted curve (in deprojected plane) and map back to image frame
    tmin, tmax = theta_fit.min(), theta_fit.max()
    theta_s = np.linspace(tmin, tmax, 2000)
    r_s = log_spiral(theta_s, a_fit, b_fit)
    Xd_s = r_s * np.cos(theta_s)
    Yd_s = r_s * np.sin(theta_s)

    # Reproject to sky plane: undo deprojection, then rotate back, then shift to center
    Yr_s = Yd_s / deproj_factor
    Xr_s = Xd_s
    c2, s2 = np.cos(PA), np.sin(PA)
    Xs_s = c2*Xr_s + (-s2)*Yr_s
    Ys_s = s2*Xr_s +  c2*Yr_s
    xs_s = Xs_s + x0
    ys_s = Ys_s + y0

    result = {
        "a": a_fit,
        "b": b_fit,
        "pitch_angle_deg": phi,
        "PA_deg_used": PA_deg_est,
        "inclination_deg": inclination_deg,
        "center": (x0, y0),
        "theta_range": (float(tmin), float(tmax)),
        "curve_xy_image": (xs_s, ys_s),  # spiral back in the original image coords
        "mask_points": (xs, ys)          # points used for the fit (pre-rotation/deprojection)
    }
    return result

# ---------- Example of plotting ----------
folder = "~/thesis/Spiral_pattern/"+sys.argv[1]
file="data_1300/RT.fits.gz"
name = folder+file

hdul = fits.open(name)
image_data = hdul[0].data
hdul.close()
image = image_data[0, 0, 0, :, :]  # select first frame

res = fit_spiral_with_deprojection(image, inclination_deg=15.0, center=None, PA_deg=None)

"""
xs_s, ys_s = res["curve_xy_image"]
#plt.imshow(image, cmap="gray", origin="lower")
#plt.plot(res["mask_points"][0], res["mask_points"][1], '.', ms=1, alpha=0.2)
plt.plot(xs_s, ys_s, '-', lw=2)
plt.title(f"Log Spiral fit: pitch={res['pitch_angle_deg']:.2f}°, PA={res['PA_deg_used']:.1f}°")
plt.show()
"""

# 1. Shift image to center
ny, nx = image.shape
x0, y0 = nx/2.0, ny/2.0
X_full = np.arange(nx) - x0
Y_full = np.arange(ny) - y0
X_grid, Y_grid = np.meshgrid(X_full, Y_full)

# 2. Rotate by -PA
PA = np.radians(res["PA_deg_used"])
c, s = np.cos(-PA), np.sin(-PA)
Xr_grid = c*X_grid - s*Y_grid
Yr_grid = s*X_grid + c*Y_grid

# 3. Deproject
i = np.radians(res["inclination_deg"])
deproj_factor = 1.0 / np.cos(i)
Yd_grid = Yr_grid * deproj_factor
Xd_grid = Xr_grid

# 4. Interpolate original image onto deprojected grid
#    Use map_coordinates for subpixel mapping
from scipy.ndimage import map_coordinates


# Coordinates for interpolation must be in (row, col) order
coords = np.array([Ys_back.ravel(), Xs_back.ravel()])

deproj_image = map_coordinates(image, coords, order=1, mode='reflect').reshape(ny, nx)

# 5. Plot
plt.figure(figsize=(8,8))
plt.imshow(deproj_image, cmap='inferno', origin='lower')
#plt.plot(Xd_s + x0, Yd_s + y0, 'c-', linewidth=2, label='Fitted spiral (deprojected)')
#plt.scatter(Xd, Yd, s=2, c='w', alpha=0.3, label='Detected pixels')
plt.legend()
plt.title("Deprojected Spiral Image")
plt.show()