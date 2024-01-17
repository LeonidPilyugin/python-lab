import numpy as np
from typing import Tuple
import pint
from uncertainties import ufloat
import math
from . import utils
from . import arr
from . import mmath

# line by LSQ method
def lsq(x, y, w = None, sl = slice(None, None)) -> Tuple[pint.Quantity, pint.Quantity]:
    if not isinstance(x, arr.Array):
        x = arr.Array(x)
    if not isinstance(y, arr.Array):
        y = arr.Array(y)
        
    x = x[sl]
    y = y[sl]
        
    # check if arguments are correct
    assert x.size == y.size
    if w is None:
        w = arr.Array(np.ones(x.size))
    assert w.size == x.size

    # sum of weights
    W = w.n.sum()
    assert W != 0

    # hepful values
    x_mean = (x * w).sum() / W
    y_mean = (y * w).sum() / W
    xy_mean = (x * y * w).sum() / W
    x2_mean = (x ** 2 * w).sum() / W
    y2_mean = (y ** 2 * w).sum() / W
    Dxy = xy_mean - x_mean * y_mean
    Dxx = x2_mean - x_mean ** 2
    Dyy = y2_mean - y_mean ** 2

    # compute parameters
    k = Dxy / Dxx
    b = y_mean - k * x_mean
    ksigma = mmath.sqrt((Dyy / Dxx - k ** 2) / (x.size - 2))[0]
    bsigma = ksigma * mmath.sqrt(x2_mean)[0]
    
    # return result
    return ufloat(k.n, ksigma.n) * k.u, ufloat(b.n, bsigma.n) * b.u


# line by chi^2 method
def chi2(x, y, sl = slice(None, None)) -> Tuple[pint.Quantity, pint.Quantity]:
    if not isinstance(x, arr.Array):
        x = arr.Array(x)
    if not isinstance(y, arr.Array):
        y = arr.Array(y)

    x = x[sl]
    y = y[sl]
    
    # check if arguments are correct
    assert x.size == y.size
    
    # compute weights
    w = arr.Array(1 / y.s ** 2)
    for item in w:
        if math.isinf(item.s) or math.isnan(item.s) or math.isinf(item.n) or math.isnan(item.n):
            w = None
            break
    
    # return result
    return lsq(x, y, w)

