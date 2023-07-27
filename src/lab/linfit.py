import numpy as np
from typing import Tuple
import pint
from uncertainties import ufloat
from . import utils
from . import arr
from . import mmath

# line by LSQ method
def lsq(x, y, w = None, sl = slice(None, None)) -> Tuple[pint.Quantity, pint.Quantity]:
    if isinstance(x, arr.Array):
        x = x.arr
    if isinstance(y, arr.Array):
        y = y.arr
    if isinstance(w, arr.Array):
        w = w.arr
        
    x = x[sl]
    y = y[sl]
        
    # check if arguments are correct
    assert len(x) == len(y)
    if w is None:
        w = np.ones(len(x))
    assert len(w) == len(x)
    
    # sum of weights
    W = w.sum()
    assert W != 0
    
    if hasattr(x[0], "units"):
        x = utils.normalize(x)
    if hasattr(y[0], "units"):
        y = utils.normalize(y)
    if hasattr(x[0], "n"):
        x = utils.nominal(x)
    if hasattr(y[0], "n"):
        y = utils.nominal(y)

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
    ksigma = mmath.sqrt(arr.Array((Dyy / Dxx - k ** 2) / (len(x) - 2))).arr[0]
    bsigma = ksigma * mmath.sqrt(arr.Array(x2_mean)).arr[0]
    
    # return result
    return ufloat(k.magnitude, ksigma.magnitude) * k.units, ufloat(b.magnitude, bsigma.magnitude) * b.units


# line by chi^2 method
def chi2(x, y, sl = slice(None, None)) -> Tuple[pint.Quantity, pint.Quantity]:
    if isinstance(x, arr.Array):
        x = x.arr
    if isinstance(y, arr.Array):
        y = y.arr
    
    x = x[sl]
    y = y[sl]
    
    # check if arguments are correct
    assert len(x) == len(y)
    
    # compute weights
    w = 1 / utils.normalize(utils.std(y)) ** 2
    
    # return result
    return lsq(x, y, w)

