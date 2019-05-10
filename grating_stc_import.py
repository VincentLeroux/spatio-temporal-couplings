import h5py


def import_h5_example_profile(filename):
    '''
    Import grating deformation profile from the example file.

    Parameters:
    -----------
    filename: str
        File path where the data is stored

    Returns:
    --------
    profile: 2D numpy.array
        2D map of the grating surface

    x: 1D numpy.array
        horizontal axis

    y: 1D numpy.array
        vertical axis
    '''
    data = {}
    with h5py.File(filename, 'r') as fid:
        for key in list(fid.keys()):
            data[key] = fid[key][...]
    profile = data['data']
    x = data['x']
    y = data['y']
    return profile, x, y
