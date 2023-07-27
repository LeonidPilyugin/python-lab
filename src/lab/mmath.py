import uncertainties.umath as umath
import numpy as np
from .unit import unit
from . import arr
import pint
import math

def _acos(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.acos(x.magnitude) * unit.rad


def acos(x) -> arr.Array:
    return arr.Array(acos.f(x.arr))

acos.f = np.vectorize(_acos, otypes=[object])


def _acosh(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.acosh(x.magnitude) * unit.dimensionless


def acosh(x) -> arr.Array:
    return arr.Array(acosh.f(x.arr))

acosh.f = np.vectorize(_acosh, otypes=[object])


def _asin(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.asin(x.magnitude) * unit.rad


def asin(x) -> arr.Array:
    return arr.Array(asin.f(x.arr))

asin.f = np.vectorize(_asin, otypes=[object])


def _atan(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.atan(x.magnitude) * unit.rad


def atan(x) -> arr.Array:
    return arr.Array(atan.f(x.arr))

atan.f = np.vectorize(_atan, otypes=[object])


def _atan2(y, x) -> pint.Quantity:
    return umath.atan2(y.to(x.units).magnitude, x.magnitude) * unit.rad


def atan2(y, x) -> arr.Array:
    return arr.Array(atan2.f(y.arr, x.arr))

atan2.f = np.vectorize(_atan2, otypes=[object, object])


def _atanh(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.atanh(x.magnitude) * unit.dimensionless


def atanh(x) -> arr.Array:
    return arr.Array(atanh.f(x.arr))

atanh.f = np.vectorize(_atanh, otypes=[object])


def _ceil(x) -> pint.Quantity:
    return umath.ceil(x.magnitude) * x.units


def ceil(x) -> arr.Array:
    return arr.Array(ceil.f(x.arr))

ceil.f = np.vectorize(_ceil, otypes=[object])


def _copysign(x, y) -> pint.Quantity:
    return umath.copysign(x.magnitude, y.magnitude) * x.units


def copysign(y, x) -> arr.Array:
    return arr.Array(copysign.f(y.arr, x.arr))

copysign.f = np.vectorize(_copysign, otypes=[object, object])


def _cos(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.cos(x.to(unit.rad).magnitude) * unit.dimensionless


def cos(x) -> arr.Array:
    return arr.Array(cos.f(x.arr))

cos.f = np.vectorize(_cos, otypes=[object, object])


def _cosh(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.cosh(x.magnitude) * unit.dimensionless


def cosh(x) -> arr.Array:
    return arr.Array(cosh.f(x.arr))

cosh.f = np.vectorize(_cosh, otypes=[object])


def _degrees(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.degrees(x.to(unit.rad).magnitude) * unit.degree


def degrees(x) -> arr.Array:
    return arr.Array(degrees.f(x.arr))

degrees.f = np.vectorize(_degrees, otypes=[object])


def _erf(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.erf(x.magnitude) * unit.dimensionless


def erf(x) -> arr.Array:
    return arr.Array(erf.f(x.arr))

erf.f = np.vectorize(_erf, otypes=[object])


def _erfc(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.erfc(x.magnitude) * unit.dimensionless


def erfc(x) -> arr.Array:
    return arr.Array(erfc.f(x.arr))

erfc.f = np.vectorize(_erfc, otypes=[object])


def _exp(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.exp(x.magnitude) * unit.dimensionless


def exp(x) -> arr.Array:
    return arr.Array(exp.f(x.arr))

exp.f = np.vectorize(_exp, otypes=[object])


def _expm1(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.expm1(x.magnitude) * unit.dimensionless


def expm1(x) -> arr.Array:
    return arr.Array(expm1.f(x.arr))

expm1.f = np.vectorize(_expm1, otypes=[object])


def _fabs(x) -> pint.Quantity:
    return umath.expm1(x.magnitude) * x.units


def fabs(x) -> arr.Array:
    return arr.Array(fabs.f(x.arr))

fabs.f = np.vectorize(_fabs, otypes=[object])


def _factorial(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.factorial(x.magnitude) * unit.dimensionless


def factorial(x) -> arr.Array:
    return arr.Array(factorial.f(x.arr))

factorial.f = np.vectorize(_factorial, otypes=[object])


def _floor(x) -> pint.Quantity:
    return umath.floor(x.magnitude) * x.units


def floor(x) -> arr.Array:
    return arr.Array(floor.f(x.arr))

floor.f = np.vectorize(_floor, otypes=[object])


def _fmod(x, y) -> pint.Quantity:
    return umath.fmod(x.magnitude, y.to(x.units).magnitude) * x.units


def fmod(x, y) -> arr.Array:
    return arr.Array(fmod.f(x.arr, y.arr))

fmod.f = np.vectorize(_fmod, otypes=[object, object])


def _fsum(args) -> pint.Quantity:
    args_ = list()
    units = args[0].units
    for c in args:
        args_.append(c.to(units))
    return umath.fsum([c.magnitude for c in args_]) * units


def _gamma(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.gamma(x.magnitude) * unit.dimensionless


def gamma(x) -> arr.Array:
    return arr.Array(gamma.f(x.arr), otypes=[object])

gamma.f = np.vectorize(_gamma)

def hypot(*coordinates) -> arr.Array:
    return sqrt(sum([c ** 2 for c in coordinates]))


def _isinf(x) -> bool:
    return umath.isinf(x.magnitude)


def isinf(x) -> arr.Array:
    return arr.Array(isinf.f(x.arr))

isinf.f = np.vectorize(_isinf, otypes=[object])


def _isnan(x) -> bool:
    return umath.isnan(x.magnitude)


def isnan(x) -> arr.Array:
    return arr.Array(isnan.f(x.arr))

isnan.f = np.vectorize(_isnan, otypes=[object])


def _lgamma(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.lgamma(x.magnitude) * unit.dimensionless


def lgamma(x) -> arr.Array:
    return arr.Array(lgamma.f(x.arr))

lgamma.f = np.vectorize(_lgamma, otypes=[object])


def _log(x, base=math.e * unit.dimensionless) -> pint.Quantity:
    assert x.dimensionless
    assert base.dimensionless
    return umath.log(x.magnitude, base.magnitude) * unit.dimensionless


def log(x, base=arr.Array(math.e * unit.dimensionless)) -> arr.Array:
    return arr.Array(log.f(x.arr, base.arr))

log.f = np.vectorize(_log, otypes=[object])


def _log10(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.log10(x.magnitude) * unit.dimensionless


def log10(x) -> arr.Array:
    return arr.Array(log10.f(x.arr))

log10.f = np.vectorize(_log10, otypes=[object])


def _log1p(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.log1p(x.magnitude) * unit.dimensionless


def log1p(x) -> arr.Array:
    return arr.Array(log1p.f(x.arr))

log1p.f = np.vectorize(_log1p, otypes=[object])


def _modf(x) -> pint.Quantity:
    units = x.units
    return umath.modf(x.magnitude) * units


def modf(x) -> arr.Array:
    return arr.Array(modf.f(x.arr))

modf.f = np.vectorize(_modf, otypes=[object])


def _pow(x, y) -> pint.Quantity:
    if isinstance(y, float) or isinstance(y, int):
        return umath.pow(x.magnitude, y) * ((1 * x.units) ** y)
    else:
        assert y.dimensionless and x.dimensionless
        return umath.pow(x.magnitude, y.magnitude) * unit.dimensionless


def pow(x, y) -> arr.Array:
    return arr.Array(pow.f(x.arr, y.arr))

pow.f = np.vectorize(_pow, otypes=[object, object])


def _radians(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.radians(x.to(unit.deg).magnitude) * unit.rad


def radians(x) -> arr.Array:
    return arr.Array(radians.f(x.arr))

radians.f = np.vectorize(_radians, otypes=[object])


def _sin(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.sin(x.to(unit.rad).magnitude) * unit.dimensionless


def sin(x) -> arr.Array:
    return arr.Array(sin.f(x.arr))

sin.f = np.vectorize(_sin, otypes=[object])


def _sinh(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.sinh(x.magnitude) * unit.dimensionless


def sinh(x) -> arr.Array:
    return arr.Array(sinh.f(x.arr))

sinh.f = np.vectorize(_sinh, otypes=[object])


def _sqrt(x) -> pint.Quantity:
    return umath.sqrt(x.magnitude) * x.units ** 0.5


def sqrt(x) -> arr.Array:
    return arr.Array(sqrt.f(x.arr))

sqrt.f = np.vectorize(_sqrt, otypes=[object])


def _tan(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.tan(x.to(unit.rad).magnitude) * unit.dimensionless


def tan(x) -> arr.Array:
    return arr.Array(tan.f(x.arr))

tan.f = np.vectorize(_tan, otypes=[object])


def _tanh(x) -> pint.Quantity:
    assert x.dimensionless
    return umath.tanh(x.magnitude) * unit.dimensionless


def tanh(x) -> arr.Array:
    return arr.Array(tanh.f(x.arr))

tanh.f = np.vectorize(_tanh, otypes=[object])


def _trunc(x) -> pint.Quantity:
    return umath.trunc(x.magnitude) * x.units


def trunc(x) -> arr.Array:
    return arr.Array(trunc.f(x.arr))

trunc.f = np.vectorize(_trunc, otypes=[object])
