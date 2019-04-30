# spatio-temporal-couplings
Python functions used to simulate the spatio-temporal couplings (STC) resulting from heat-induced deformation of compressor gratings.

* grating_stc_main.py contains the main script which defines the simulation parameter, run the simulation and display some of the output parameters.
* grating_stc_function.py contains msot of the functions used to construct the beam, or derive its main properties.
* Example files of gratings deformations are found in the data folder. They are h5 files with the following datasets:
 * 'x': x axis, in m
 * 'y': y axis, in m
 * 'data': deformation map, in m
