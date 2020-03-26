# spatio-temporal-couplings
Python functions used to simulate the spatio-temporal couplings (STC) resulting from heat-induced deformation of compressor gratings.

To run it, just clone or copy the repository (3 .py files, and the data folder with its contents), and run the grating_stc_main.py.

If you use the code, please consider to cite the following reference: V. Leroux et al., Opt. Express 28, 8257-8265 (2020).

* grating_stc_main.py contains the main script which defines the simulation parameter, run the simulation and display some of the output parameters.
* grating_stc_function.py contains most of the functions used to construct the beam, or derive its main properties.
* grating_stc_import.py contains the function used to import the data into the main script. User-specific import functions can be added here and imported in the main script to load the data.
* Example files of gratings deformations are found in the data folder. They are h5 files with the following datasets:
  * 'x': x axis, in m
  * 'y': y axis, in m
  * 'data': deformation map, in m


  If you use the code, please consider to cite the following reference: [V. Leroux et al., Opt. Express 28, 8257-8265 (2020)](https://doi.org/10.1364/OE.386112)
