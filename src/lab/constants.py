from .unit import unit as u
from uncertainties import ufloat
import math



# math
pi = math.pi * u.dimensionless
e = math.e * u.dimensionless

# fundamental
c = 299_792_458 * u.m / u.s
G = ufloat(6.674_30e-11, 0.00015e-11) * u.m ** 3 / u.kg / u.s ** 2
h = 6.626_070_15e-34 * u.J * u.s
e = 1.602_176_634e-19 * u.C
kB = 1.380_649e-23 * u.J / u.K

# contact
Na = 6.022_140_76e23 / u.mol
alpha = ufloat(7.297_352_5693e-3, 0.0000000011e-3) * u.dimensionless
eps0 = ufloat(8.854_187_8128e-12, 0.0000000013e-12) * u.F / u.m
aem = ufloat(1.660_539_066_60e-27, 0.00000000050e-27) * u.kg
eV = e * u.V
R = kB * Na

# em
mu0 = 1 / eps0 / c ** 2
Z0 = mu0 * c
k = 1 / 4 / pi / eps0

# masses
me = ufloat(9.109_383_7015e-31, 0.0000000028e-31) * u.kg
mp = ufloat(1.672_621_923_69e-27, 0.00000000051e-27) * u.kg
mn = ufloat(1.674_927_498_04e-27, 0.00000000095e-27) * u.kg

# other
g = ufloat(9.8154, 0.0001) * u.m / u.s ** 2

