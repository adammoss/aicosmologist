# Dataset Description

These are found in the `data` directory.

latin_hypercube_params.txt: 

Lists the cosmological parameters used in each of the 2000 simulations, of the form:

#Omega_m                 Omega_b                  h                        n_s                      sigma_8
1.755000000000000171e-01 6.681000000000000827e-02 7.737000000000000544e-01 8.849000000000000199e-01 6.641000000000000236e-01
2.139000000000000346e-01 5.557000000000000828e-02 8.599000000000001087e-01 9.785000000000000364e-01 8.618999999999998884e-01


latin_hypercube_3D.npy: 

(2000, 64, 64, 64) numpy array of 2000 simulations of the CDM density field, on a 64x64x64 grid, with a box size of 1 Gpc/h.