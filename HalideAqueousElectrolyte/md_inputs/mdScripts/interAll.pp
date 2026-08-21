# water is SPC/E here
set type 1 charge  0.4238 # Type 1 is H
set type 2 charge -0.8476 # Type 2 is 0

# Halides
set type 3 charge -1.0 # F
set type 4 charge -1.0 # Cl
set type 5 charge -1.0 # Br
set type 6 charge -1.0 # I

# Alkali
set type 7 charge 1.0 # Li
set type 8 charge 1.0 # Na
set type 9 charge 1.0 # K
set type 10 charge 1.0 # Rb
set type 11 charge 1.0 # Cs


# Defining the pair style. In this case is a LJ with cutoff and Coulombic interaction is treated not
# with a cutoff but with PPPM
pair_style lj/cut/coul/long 10.0
kspace_style pppm 1.0e-5
#kspace_modify gewald 0.1

pair_coeff 1 1 0.0000 1.00000 # H
pair_coeff 2 2 0.1553 3.166 # O

pair_coeff 3 3 0.0074005 4.022 # F
pair_coeff 4 4 0.012785 4.830 # Cl
pair_coeff 5 5 0.0269586 4.902 # Br
pair_coeff 6 6 0.0427845 5.201 # I

pair_coeff 7 7 0.3367344 1.409 # Li
pair_coeff 8 8 0.3526418 2.160 # Na
pair_coeff 9 9 0.4297054 2.838 # K
pair_coeff 10 10 0.4451036 3.095 # Rb
pair_coeff 11 11 0.0898565 3.601 # Cs

bond_style harmonic
bond_coeff 1 1000.0 1.0
angle_style harmonic
angle_coeff 1 100.0 109.47

pair_modify tail yes mix arithmetic
