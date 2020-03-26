import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import h5py
from scipy.interpolate import interp2d
import grating_stc_functions as stc
from grating_stc_import import import_h5_example_profile
from time import time

script_t0 = time()

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
tau_chirped = 200e-12  # Chirped pulse duration, in s


################### Compute remaining parameters ###################

# Frequency axis
om, om0, Dom, Nom, k0 = stc.build_omega_array(
    l0=l0, Nom=Nom, bandwidth=rel_bandwidth, omega_span=freq_axis_span)
tau_0 = 4 * np.log(2) / Dom  # Fourier limited pulse duration for Gaussian shape, in s
phi2 = -tau_chirped / Dom  # GDD to compensate

# Get diffracted angles axis and remove imaginary rays
beta = stc.grating_formula(alpha, om, gr_spacing)  # Diffracted angles, in rad
om = om[np.isfinite(beta)]  # Remove imaginary rays
k = om / c  # Wave vector amplitude, in rad/m
Nom = om.size  # Size of frequency axis, without imaginary rays
beta = beta[np.isfinite(beta)]  # Remove imaginary rays
beta0 = stc.grating_formula(alpha, om0, gr_spacing)  # Angle chief ray
idx_om0 = np.argmin(np.abs(om - om0)) # index of central frequency

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
L_dist = stc.grating_distance(beta0, om0, phi2)  # Normal to the gratings, in m
dx = (np.tan(beta0) - np.tan(beta)) * \
    np.cos(alpha) * L_dist  # Hor. dispersion, in m


################### Grating deformation ###################
# Can be replaced by measured or simulated data

# Import gratings deformation and put the data in a dictionary
g1 = {}
g2 = {}
g1['data'], g1['x'], g1['y'] = import_h5_example_profile(
    './data/G1_deformation_example.h5')
g2['data'], g2['x'], g2['y'] = import_h5_example_profile(
    './data/G2_deformation_example.h5')

# Interpolate on the beam coordinates.
g1_interp = interp2d(g1['x'] * np.cos(alpha), g1['y'],
                     g1['data'], kind='cubic', fill_value=0)
g2_interp = interp2d(g2['x'] * np.cos(alpha), g2['y'],
                     g2['data'], kind='cubic', fill_value=0)

# Calculate deformation at each pass
phi_pass1 = g1_interp(x, y + H1 / 2)
phi_pass2 = np.zeros([Ny, Nx, Nom])
phi_pass3 = np.zeros([Ny, Nx, Nom])
for idx in np.arange(Nom):
    phi_pass2[:, :, idx] = g2_interp(x + dx[idx], y + H1 / 2)
    phi_pass3[:, :, idx] = np.flipud(g2_interp(x + dx[idx], y - H1 / 2))
phi_pass4 = np.flipud(g1_interp(x, y - H1 / 2))
# 3rd and 4th passes are flipped (up-down) by the roof mirror

# Total wavefront accumulated from the compressor
phi_comp = phi_pass1[:, :, None] + phi_pass2 + \
    phi_pass3 + phi_pass4[:, :, None]

# Project deformation to the wavefront plane
phi_comp *= stc.deformation_scaling(alpha, beta)

################### Build Near field ###################

E_nf = stc.gauss2D(x=X, y=Y, fwhmx=2 * w0, fwhmy=2 * w0, order=6) # Super-Gaussian spatial near E-field
E_om = stc.gauss1D(x=om, fwhm=Dom, x0=om0, order=1) # Gaussian spectral near E-field

# Spatio-spectral near field with phase
E_nf_om = E_nf[:, :, None] * E_om[None, None, :] * \
    np.exp(1j * k[None, None, :] * phi_comp)

# Get GDD of the center of the beam
phi2_nf, _, _ = stc.get_stc_coeff(E_nf_om, w0, om0, Dom / 2, x, om, level=0.5)

# Get curvature (divergence) of central wavelength
_, rad_curv_x, _, _ = stc.divergence_from_phase(
    np.unwrap(np.angle(E_nf_om[Ny // 2, :, idx_om0])), x, k0, w0)
_, rad_curv_y, _, _ = stc.divergence_from_phase(
    np.unwrap(np.angle(E_nf_om[:, Nx // 2, idx_om0])), y, k0, w0)
rad_curv = (rad_curv_x + rad_curv_y) / 2


# Remove GDD and curvature for propagation
E_nf_om = E_nf_om * \
    np.exp(1j * phi2_nf / 2 * (om - om0)**2)[None, None, :] * \
    np.exp(1j * k[None, None, :] / (2 * rad_curv) * R[:, :, None]**2)

# Spatio-spectral far field
E_ff_om = np.fft.fftshift(np.fft.fft2(E_nf_om, axes=(0, 1)), axes=(0, 1))

# Remove Group Delay to center the trace in time axis
on_axis_phase = np.unwrap(
    np.angle(E_nf_om[Ny // 2, Nx // 2, np.abs(om - om0) <= Dom / 2]))
mean_gd = np.mean(np.diff(on_axis_phase)) / (om[1] - om[0])
gd_comp = np.exp(-1j * mean_gd * (om - om0))
E_ff_om = E_ff_om * gd_comp[None, None, :]

# Spatio-temporal far field
E_ff_t = np.fft.fftshift(np.fft.fft(E_ff_om, axis=-1), axes=-1)

################### Beam properties ###################

# Intensity profile
I_ff_t = np.abs(E_ff_t)**2

# Pulse duration
tau = stc.get_fwhm(np.sum(I_ff_t, axis=(0, 1)), interpolation_factor=10) * dt
# Waist in pixels
_, _, wx, wy = stc.get_moments(np.sum(I_ff_t, axis=-1))
# PFT
pft_ff = stc.get_pft(E_ff_t, xf, t, level=0.5)
# GDD, Spatial chirp, angular chirp
phi2_ff, zeta_ff, beta_ff = stc.get_stc_coeff(
    E_ff_om, wx, om0, Dom / 2, xf, om, level=0.01)
# Waist in m
wx *= dxf
wy *= dyf

################### Display ###################

print('\nBeam parameters in focus:')
print('-------------------------\n')
print('Pulse duration: {:.2f} fs'.format(tau * 1e15))
print('Waist x: {:.2f} µm'.format(wx * 1e6))
print('Waist y: {:.2f} µm'.format(wy * 1e6))
print('PFT: {:.2f} fs/µm'.format(pft_ff * 1e9))
print('GDD: {:.2f} fs2'.format(phi2_ff * 1e30))
print('Spatial chirp: {:.2f} mm/(rad/fs)'.format(zeta_ff * 1e18))
print('Angular chirp: {:.2f} mrad/(rad/fs)'.format(beta_ff * 1e18))

print('\nSimulation time: {:.2f} sec'.format(time() - script_t0))

plt.subplot(1, 3, 1)
plt.imshow(np.sum(I_ff_t, axis=0), cmap='YlGnBu_r', aspect='equal',
           extent=[t[0] * 1e15, t[-1] * 1e15, xf[0] * 1e6, xf[-1] * 1e6])
plt.xlim(-150, 150)
plt.ylim(-150, 150)
plt.xlabel('t (fs)')
plt.ylabel('x (µm)')
plt.title('T-X')

plt.subplot(1, 3, 2)
plt.imshow(np.sum(I_ff_t, axis=1), cmap='YlGnBu_r', aspect='equal',
           extent=[t[0] * 1e15, t[-1] * 1e15, yf[0] * 1e6, yf[-1] * 1e6])
plt.xlim(-150, 150)
plt.ylim(-150, 150)
plt.xlabel('t (fs)')
plt.ylabel('y (µm)')
plt.title('T-Y')

plt.subplot(1, 3, 3)
plt.imshow(np.sum(I_ff_t, axis=2), cmap='YlGnBu_r', aspect='equal',
           extent=[xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6])
plt.xlim(-150, 150)
plt.ylim(-150, 150)
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('X-Y')

plt.tight_layout()
plt.savefig('beam_profiles.png', bbox_inches='tight')
plt.show()
