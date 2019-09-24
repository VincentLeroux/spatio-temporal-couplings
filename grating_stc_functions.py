import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d


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
    if isinstance(om, float):
        om = np.array([om])
    sin_beta = np.sin(alpha) - np.sign(alpha) * 2 * np.pi * c / om / gd
    sin_beta[np.abs(sin_beta) > 1] = np.nan
    if sin_beta.size == 1:
        sin_beta = sin_beta[0]
    return np.arcsin(sin_beta)


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

def deformation_scaling(alpha, beta):
	'''
	Calculate the scaling factor that is needed to project the 
	deformation profile of the grating to the plane of the 
	wavefront

	Parameters:
	-----------
	alpha: float
        Input angle, in radians

	beta: float, numpy.array
        Diffracted angle, in radians
	'''
	return (1/np.cos(alpha) + 1/np.cos(beta))


def divergence_from_phase(phase, x, k, hwhm, cut=0.5):
    '''
    Calculate beam divergence, radius of curvature
    and tilt from a 1D spatial phase.

    Returns:
    --------
    divergence:
        beam divergence in rad

    radius_curvature:
        radius of curvature in m

    residual:
        residuals of polynomial fit, to assess fit quality

    slope:
        Tilt of the phase, in rad

    Parameters:
    -----------
    phase: numpy.array
        spatial phase of the beam

    x: numpy.array
        transverse axis

    k: float
        wave vector amplitude

    hwhm: float
        half-width at half maximum of the beam.

    cut: float, optional
        the phase above cut*hwhm is ignored
    '''
    idx_cut = np.abs(x) <= cut * hwhm
    x_cut = x[idx_cut]
    phase_fit, residual, _, _, _ = np.polyfit(
        x_cut, phase[idx_cut], 4, full=True)
    radius_curvature = -k / (2 * phase_fit[-3])
    divergence = np.arcsin(hwhm / radius_curvature)
    residual = residual[0]
    slope = phase_fit[-2]
    return divergence, radius_curvature, residual, slope


def gauss2D(x, y, fwhmx, fwhmy, x0=0, y0=0, offset=0, order=1, int_FWHM=True):
    '''
    Define a (super-)Gaussian 2D beam.

    Parameters:
    -----------
    x: float 2D np.array
        Horizontal axis of the Gaussian

    y: float 2D np.array
        Vertical axis of the Gaussian

    fwhmx: float
        Horizontal Full Width at Half Maximum

    fwhmy: float
        Vertical Full Width at Half Maximum

    x0: float, optional
        Horizontal center position of the Gaussian

    y0: float, optional
        Vertical center position of the Gaussian

    offset: float, optional
        Amplitude offset of the Gaussian

    order: int, optional
        order of the super-Gaussian function.
        Defined as: exp( - ( x**2 + y**2 )**order )

    int_FWHM: boolean, optional
        If True, the FWHM is the FWHM of the
        square of the Gaussian (intensity).
        If False, it is the FWHM of the Gaussian
        directly (electric field).
    '''
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    gauss = np.exp(-np.log(2) * coeff *
                   ((2 * (x - x0) / fwhmx) ** 2 +
                    (2 * (y - y0) / fwhmy)**2)**order) + offset
    return gauss


def gauss1D(x, fwhm, x0=0, offset=0, order=1, int_FWHM=True):
    '''
    Define a (super-)Gaussian 1D beam.
    Identical to gauss2D but in 1D.

    Parameters:
    -----------
    x: float 1D np.array
        Axis of the Gaussian

    fwhm: float
        Full Width at Half Maximum

    x0: float, optional
        Center position of the Gaussian

    offset: float, optional
        Amplitude offset of the Gaussian

    order: int, optional
        order of the super-Gaussian function.
        Defined as: exp( - ( x**2 )**order )

    int_FWHM: boolean, optional
        If True, the FWHM is the FWHM of the
        square of the Gaussian (intensity).
        If False, it is the FWHM of the Gaussian
        directly (electric field).
    '''
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    gauss = np.exp(-np.log(2) * coeff *
                   ((2 * (x - x0) / fwhm)**2)**order) + offset
    return gauss


def get_stc_coeff(E_field_xyom, w0, om0, Dom, x, om, level=0.5):
    '''
    Calculate first order STC coefficients and temporal chirp

    Returns:
    --------
    phi2:
        temporal chirp in the center of the beam, in s2

    zeta:
        spatial chirp in x axis, in m.s/rad

    beta:
        angular chirp in x axis, in rad.s/rad

    Parameters:
    -----------
    E_field_xyom: 3D numpy.array
        data aranged in following order (y, x, omega)

    w0: float
        beam transverse waist in m,
        used to limit calculation area

    om0: float
        central frequency in rad/s

    Dom: float
        half bandwith in rad/s,
        used to limit calculation area

    x: 1D numpy.array
        x axis in m

    om: 1D numpy.array
        angular frequency axis in rad/s

    level: float
        signal amplitude below which data
        is ignored for spatial chirp
    '''

    k0 = om0 / c
    # indexes
    idx_y = np.argmax(np.sum(np.sum(E_field_xyom, axis=-1), axis=-1))
    idx_om = np.nonzero(np.abs(om - om0) < Dom)[0]
    idx_x_arr = np.abs(x) < w0
    idx_x = int(np.round(np.mean(idx_x_arr)))

    # Spatial chirp
    om_zeta, x_zeta = image_max_loc(np.abs(E_field_xyom).sum(
        0), level=level, axis=0, x0=om, x1=x)
    zeta = -np.polyfit(om_zeta, x_zeta, 1)[-2]

    # Temporal chirp
    phi2 = np.polyfit(om[idx_om], np.unwrap(
        np.angle(E_field_xyom[idx_y, idx_x, idx_om])), 2)[-3] * 2

    # Angular chirp
    slope = np.zeros(idx_om.size)
    for i, ii in enumerate(idx_om):
        slope[i] = np.polyfit(x[idx_x_arr], np.unwrap(
            np.angle(E_field_xyom[idx_y, idx_x_arr, ii])), 1)[-2]
    beta = -np.polyfit(om[idx_om], slope, 1)[-2] / k0

    return phi2, zeta, beta


def get_pft(E_field_xyt, x, t, level=0.5):
    '''
    Calculate the PFT along the x-axis of a 3D E-field

    Parameters:
    -----------
    E_field_xyt: 3D numpy.array
        data aranged in following order (y, x, t)

    x: 1D numpy.array
        x axis in m

    t: 1D numpy.array
        time axis in s

    level: float
        signal amplitude below which data is ignored
    '''
    x_p, t_p = image_max_loc(np.abs(E_field_xyt).sum(0).T,
                             level=level, axis=0, x0=x, x1=t)
    p = np.polyfit(x_p, t_p, 1)[-2]
    return p


def image_max_loc(image, level=0.5, cut=None, axis=0, x0=None, x1=None):
    '''
    Gives the position of the maxima of an image along a given axis

    Returns:
    --------
    x0_max: numpy.array
        index of maxima on the first axis

    x1_max: numpy.array
        index of maxima on the second axis

    Parameters:
    -----------
    image: 2D numpy.array
        input image

    level: float or None, optional
        signal amplitude below which data is ignored

    cut: float or None, optional
        if level is None, reduces the data range by
        cutting the axis at the cut value

    axis: 0 or 1, optional
        Axis along which the maxima are found

    x0: numpy.array or None, optional
        Position vector of the first axis

    x1: numpy.array or None, optional
        Position vector of the second axis
    '''

    if x0 is None:
        x0 = np.arange(image.shape[0])
    if x1 is None:
        x1 = np.arange(image.shape[1])

    if level is not None:
        idx = np.sum(image, axis=axis) >= level * \
            np.max(np.sum(image, axis=axis))
    elif axis == 0:
        idx = np.abs(x0) <= cut
    elif axis == 1:
        idx = np.abs(x1) <= cut

    if axis == 0:
        x0_max = x0[idx]
        x1_max = (np.argmax(image[:, idx], axis=0) -
                  x1.size / 2) * (x1[1] - x1[0])
    elif axis == 1:
        x1_max = x1[idx]
        x0_max = (np.argmax(image[idx, :], axis=1) -
                  x0.size / 2) * (x0[1] - x0[0])

    return x0_max, x1_max


def get_moments(image):
    '''
    Compute image centroid and statistical waist
    from the intensity distribution.
    Returns x0, y0, wx, wy

    Parameters:
    -----------
    image: 2D numpy array
    '''

    # Build axes in pixels
    ny, nx = image.shape
    x, y = np.arange(nx), np.arange(ny)
    X, Y = np.meshgrid(x, y)
    # Zeroth moment
    c0 = np.sum(image)
    # First moments
    cx = np.sum(X * image) / c0
    cy = np.sum(Y * image) / c0
    # Second centered moments
    sx2 = np.sum((X - cx)**2 * image) / c0
    sy2 = np.sum((Y - cy)**2 * image) / c0
    return cx, cy, 2 * np.sqrt(sx2), 2 * np.sqrt(sy2)


def get_fwhm(intensity, interpolation_factor=1, kind='cubic'):
    '''
    Get the Full Width at Half Maximum of the 1D intensity distribution

    Parameters:
    -----------
    intensity: 1D numpy array
        intensity distribution

    interpolation_factor: int, optional
        Interpolate the data for a more accurate calculation

    kind : str or int, optional
        Specifies the kind of interpolation as a string.
        See documentation of scipy.interp1d
    '''
    position = np.arange(intensity.size)
    pos_i = np.linspace(np.min(position), np.max(position),
                        interpolation_factor * position.size)
    inten_i = interp1d(position[:], intensity[:], kind=kind)
    idx = (inten_i(pos_i) >= np.max(inten_i(pos_i)) * 0.5).nonzero()[0]
    return pos_i[idx[-1] + 1] - pos_i[idx[0]]
