import numpy as np
from scipy.constants import c


def build_omega_array(l0=8e-7, Nom=256, bandwidth=0.05, omega_span=20):
    '''
    Build angular frequency array from central wavelength and bandwidth.
    Negative frequencies are removed.

    Parameters:
    -----------
    l0: float, optional
        Central wavelength, in metres
        Default is 800 nm

    Nom: integer, optional
        Number of points in axis
        Default is 256

    bandwidth: float, optional
        Relative bandwidth
        Default is 5%, which is 40 nm at 800 nm

    omega_span: float, optional
        Extent of the frequency axis.
        Can be tuned down to increase resolution
        in frequency domain, or up to increase
        resolution in time domain.

    Outputs:
    --------
    om: numpy.array
        Angular frequency array, in rad/s

    om0: float
        Central angular frequency, in rad/s

    Dom: float
        Absolute angular frequency bandwidth, in rad/s

    Nom: integer
        Number of points

    k0: float
        Central wave vector amplitude in rad/m

    '''
    om0 = 2 * np.pi * c / l0
    k0 = om0 / c
    Dom = om0 * bandwidth
    om = np.linspace(-omega_span * Dom, omega_span * Dom, Nom) + om0
    om = om[om > 0]
    Nom = om.size
    return om, om0, Dom, Nom, k0


def grating_formula(alpha, om, gd=1 / 1.48e6):
    '''
    Get first diffracted order angle

    Parameters:
    -----------
    alpha: float
        Input angle, in radians

    om: float, numpy.array
        Angular frequency, in rad/s

    gd: float, optional
        Grating groove spacing, in m.
        Inverse of the groove density in lines/m.
        Default groove density is 1480 lines/mm
    '''
    return np.arcsin(np.sin(alpha) - np.sign(alpha) * 2 * np.pi * c / om / gd)


def grating_distance(beta, om, phi2, gd=1 / 1.48e6):
    '''
    Get distance between gratings (normal to the grating surface)
    to compensate GDD

    Parameters:
    -----------
    beta: float, numpy.array
        Diffracted angle, in radians

    om: float, numpy.array
        Angular frequency, in rad/s. Dimension should match with beta.

    phi2: float
        Group delay dispersion to compensate

    gd: float, optional
        Grating groove spacing, in m.
        Inverse of the groove density in lines/m.
        Default groove density is 1480 lines/mm
    '''
    return gd**2 * phi2 * (np.cos(beta) * om)**3 / (8 * np.pi**2 * c)


def divergence_from_phase(phase, x, k, hwhm):
    '''
    Calculate beam divergence, radius of curvature
    and tilt from a 1D spatial phase.

    Parameters:
    -----------
    phase: numpy.array
        spatial phase of the beam

    x: numpy.array
        transverse axis

    k: float
        wave vector amplitude

    hwhm: float
        Half-width at half maximum of the beam.
    '''
    phase_fit, residual, _, _, _ = np.polyfit(x, phase, 4, full=True)
    radius_curvature = -k / (2 * phase_fit[-3])
    divergence = np.arcsin(hwhm / radius_curvature)
    return divergence, radius_curvature, residual[0], phase_fit[-2]
