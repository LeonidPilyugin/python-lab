from typing import Any
import numpy as np
import pint
from . import utils

class Array:
    _f = np.vectorize(lambda x, u: x.ito(u), otypes=[object, object])
    
    def __init__(self, arr):
        if isinstance(arr, np.ndarray):
            self.arr = utils.normalize(arr)
        elif isinstance(arr, Array):
            self.arr = arr.arr
        else:
            self.arr = np.asarray([arr], dtype=object)
        
    
    def __add__(self, other):
        return Array(self.arr + Array(other).arr)
    
    
    __radd__ = __add__
    
    
    def __sub__(self, other):
        return Array(self.arr - Array(other).arr)
    
    
    __rsub__ = __sub__
        
        
    def __mul__(self, other):
        return Array(self.arr * Array(other).arr)
    
    
    __rmul__ = __mul__
    
    
    def __truediv__(self, other):
        return Array(self.arr / Array(other).arr)
    
    
    __rtruediv__ = __truediv__
    
    
    def __pow__(self, other):
        return Array(self.arr ** Array(other).arr)
    
    
    __rpow__ = __pow__
    
    
    def ito(self, unit):
        for a in self.arr:
            a.ito(unit)
        return self

    def set_err(self, func):
        self.arr = utils.create_measure(self.arr, func(self.arr))
        return self
    
    
    @property
    def u(self):
        return self.arr[0].units
    
    
    @property
    def m(self):
        return self / (1 * self.u)
    
    
    def __getitem__(self, key):
        return self.arr.__getitem__(key)
    
    
    def __setitem__(self, key, value):
        return self.arr.__setitem__(key, value)
    
    
    def __str__(self):
        return self.arr.__str__()
    
    
    def __repr__(self):
        return self.arr.__repr__()
    
    
    def __getattr__(self, attr):
        return self.arr.__getattribute__(attr)
