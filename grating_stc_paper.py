import numpy as np
import matplotlib.plyplot as plt
from scipy.constants import c
from grating_stc_paper_functions import build_omega_array, grating_formula, grating_distance, divergence_from_phase


################### Simulation input parameters ###################

Nom = 512  # Number of points in frequency axis
Nx = 256  # Number of points in horizontal axis
Ny = 256  # Number of points in vertical axis

# Wavelength parameters
l0 = 800e-9  # Central wavelength, in m
rel_bandwidth = 0.04  # Relative bandwidth
freq_axis_span = 20
Dl = l0 * rel_bandwidth  # Absolute bandwith, in m

# Gratings parameters
gr_spacing = 1 / 1.48e6  # Groove spacing, in m
L1 = 0.21 / 2  # half length of first grating, in m
H1 = 0.2 / 2  # half height of first grating, in m
L2 = 0.3 / 2  # half length of second grating, in m
H2 = H1  # half height of second grating, in m
alpha = np.deg2rad(51.4)  # Incidence angle, in rad

# Spatial parameters
Lx = 0.3  # half horizontal box size, in m
Ly = Lx  # half vertical box size, in m
focal_length = 2  # Focal length, in m
w0 = 0.035  # Laser near field waist, in m
tau_chirped = 200e-12  # Chirped pulse duration


################### Compute remaining parameters ###################

# Frequency axis
om, om0, Dom, Nom, k0 = build_omega_array(
    l0=l0, Nom=Nom, bandwidth=rel_bandwidth, omega_span=freq_axis_span)
tau_0 = 4 * np.log(2) / Dom  # Fourier limited pulse duration, in s
phi2 = -tau_chirped / Dom  # GDD to compensate

# Get diffracted angles axis and remove imaginary rays
beta = grating_formula(alpha, om, gr_spacing)  # Difracted angles, in rad
om = om[np.isfinite(beta)]  # Remove imaginary rays
k = om / c  # Wave vector amplitude, in rad/m
Nom = om.size  # Size of frequency axis, without imaginary rays
beta = beta[np.isfinite(beta)]  # Remove imaginary rays
beta0 = grating_formula(alpha, om0, gr_spacing)  # Angle chief ray

# Time axis (for Fourier transform)
dt = 1 / np.ptp(om / 2 / np.pi)  # Spacing time axis, in s
t = dt * np.arange(Nom)  # Time axis, in s
t -= np.mean(t)  # Centering time axis

# Transverse axes near field
x = np.linspace(-1, 1, Nx) * Lx  # Horizontal axis, in s
y = np.linspace(-1, 1, Ny) * Lx  # Vertical axis, in s
X, Y = np.meshgrid(x, y)  # 2D grid
R = np.sqrt(X**2 + Y**2)  # Polar coordinates

# Transverse axes far field
dxf = l0 * focal_length / np.ptp(x)  # Spacing, in m
dyf = l0 * focal_length / np.ptp(y)  # Spacing, in m
xf = dxf * np.arange(Nx)  # Horizontal axis, in m
xf -= np.mean(xf)  # Centering
yf = dyf * np.arange(Ny)  # Vertical axis, in m
yf -= np.mean(yf)  # Centering

# Grating separation, and horizontal offset on G2
L_dist = grating_distance(beta0, om0, phi2)  # Normal to the gratings, in m
dx = (np.tan(beta0) - np.tan(beta)) * \
    np.cos(alpha) * L_dist  # Hor. dispersion, in m


################### Grating deformation ###################
# Can be replaced by measured or simulated data
