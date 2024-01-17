import uncertainties.umath as umath
import numpy as np
from .unit import unit
from . import arr
import pint
import math

def _one_arg_func(umath_func, input_dim=None, output_dim=lambda _x: _x):
    def inner(x):
        x = arr.Array(x)
        if not input_dim is None:
            assert x.u == input_dim

        res = x.copy()
        for i in range(res.size):
            res[i] = umath_func(res[i].m) * (output_dim if not callable(output_dim) else output_dim(x.u))

        return res

    def inner2(x):
        return inner(x)[0]

    return inner, inner2

acos, _acos = _one_arg_func(umath.acos, unit.dimensionless, unit.rad)
acosh, _acosh = _one_arg_func(umath.acosh, unit.dimensionless, unit.dimensionless)
asin, _asin = _one_arg_func(umath.asin, unit.dimensionless, unit.rad)
atan, _atan = _one_arg_func(umath.atan, unit.dimensionless, unit.rad)
atanh, _atanh = _one_arg_func(umath.atanh, unit.dimensionless, unit.dimensionless)
cos, _cos = _one_arg_func(umath.cos, unit.dimensionless, unit.dimensionless)
cosh, _cosh = _one_arg_func(umath.cosh, unit.dimensionless, unit.dimensionless)
erf, _erf = _one_arg_func(umath.erf, unit.dimensionless, unit.dimensionless)
erfc, _erfc = _one_arg_func(umath.erfc, unit.dimensionless, unit.dimensionless)
exp, _exp = _one_arg_func(umath.exp, unit.dimensionless, unit.dimensionless)
expm1, _expm1 = _one_arg_func(umath.expm1, unit.dimensionless, unit.dimensionless)
fabs, _fabs = _one_arg_func(umath.fabs)
gamma, _gamma = _one_arg_func(umath.gamma, unit.dimensionless, unit.dimensionless)
lgamma, _lgamma = _one_arg_func(umath.lgamma, unit.dimensionless, unit.dimensionless)
log10, _log10 = _one_arg_func(umath.log10, unit.dimensionless, unit.dimensionless)
log, _log = _one_arg_func(umath.log, unit.dimensionless, unit.dimensionless)
log1p, _log1p = _one_arg_func(umath.log1p, unit.dimensionless, unit.dimensionless)
sin, _sin = _one_arg_func(umath.sin, unit.dimensionless, unit.dimensionless)
sinh, _sinh = _one_arg_func(umath.sinh, unit.dimensionless, unit.dimensionless)
sqrt, _sqrt = _one_arg_func(umath.sqrt, None, lambda x: x ** 0.5)
tan, _tan = _one_arg_func(umath.tan, unit.dimensionless, unit.dimensionless)
tanh, _tanh = _one_arg_func(umath.tanh, unit.dimensionless, unit.dimensionless)

