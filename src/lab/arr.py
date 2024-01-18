from typing import Any
from collections.abc import Iterable
import numpy as np
from uncertainties import ufloat
import pint
from . import utils, unit as un


class Array:
    @staticmethod
    def toac(val):
        """Convert val to Array-compatible type
        """
        old_val = val

        if isinstance(val, Array):
            val = val.arr

        if not isinstance(val, Iterable) and not isinstance(val, pint.Quantity):
            val = [val]

        # convert val to ndarray if possible
        if isinstance(val, list) or isinstance(val, tuple):
            val = np.asarray(val, dtype=object)
        if isinstance(val, pint.Quantity):
            if hasattr(val.m, "copy"):
                vu = val.u
                val = val.m.copy().astype(object)
                for i in range(len(val)):
                    val[i] *= vu
            else:
                val = np.array([val], dtype=object)
        if not isinstance(val, np.ndarray):
            raise ValueError(f"Unknown value type {type(old_val)}")

        val = val.astype(object)
        assert(isinstance(val, np.ndarray))

        # convert units to quantities
        first_val = val[0]
        if not isinstance(first_val, pint.Quantity):
            for i in range(len(val)):
                val[i] *= un.unit("dimensionless")
            first_val = val[0]

        assert(all([isinstance(v, pint.Quantity) for v in val]))

        # convet all values to first element units
        unit = first_val.u
        for i in range(1, len(val)):
            val[i] = val[i].to(unit)

        # add uncertainty
        if not (hasattr(first_val.m, "s") and hasattr(first_val.m, "n")):
            for i in range(len(val)):
                val[i] = ufloat(val[i].m, 0) * unit

        assert(all([hasattr(v.m, "s") and hasattr(v.m, "n") for v in val]))

        return val
    
    def __init__(self, arr):
        self.arr = Array.toac(arr)
        
    
    def __add__(self, other):
        return Array(self.arr + Array.toac(other))
    
    
    def __sub__(self, other):
        return Array(self.arr - Array.toac(other))
    
    
    def __rsub__(self, other):
        return Array(Array.toac(other) - self.arr)
        
        
    def __mul__(self, other):
        if isinstance(other, pint.Quantity):
            res = self.arr.copy()
            for i in range(len(res)):
                res[i] = res[i] * other
            return Array(res)

        return Array(self.arr * Array.toac(other))
    
    
    def __truediv__(self, other):
        if isinstance(other, pint.Quantity):
            res = self.arr.copy()
            for i in range(len(res)):
                res[i] = res[i] / other
            return Array(res)

        return Array(self.arr / Array.toac(other))
    
    
    def __pow__(self, other):
        if isinstance(other, Array) and other.size == 1:
            other = other[0].m.n
        if isinstance(other, int) or isinstance(other, float):
            res = self.arr.copy()
            for i in range(len(res)):
                res[i] = res[i] ** other
            return Array(res)
        else:
            raise ValueError(f"Argument must be int or float, not {type(other)}")
    
    
    def ito(self, unit):
        for a in self.arr:
            a.ito(unit)
        return self

    def apply(self, func):
        for i in range(len(self.arr)):
            self.arr[i] = func(self.arr[i])
    

    @property
    def dimensionless(self):
        return self.arr[0].u == un.unit.dimensionless

    
    @property
    def u(self):
        """Return units"""
        return self.arr[0].u
    
    
    @property
    def m(self):
        """Return numpy array of magnitudes"""
        return np.array([v.m for v in self.arr], dtype=object)


    @property
    def s(self):
        """Return numpy array of stds"""
        return np.array([v.m.s for v in self.arr])


    @property
    def n(self):
        """Return numpy array of nominals"""
        return np.array([v.m.n for v in self.arr])
    
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.arr.__getitem__(key)

        if not isinstance(key, tuple):
            key = key,

        res = None
        for k in key:
            val = self.arr.__getitem__(k)
            if isinstance(k, int):
                val = val,
            res = Array(val) if res is None else Array.concat(*res, *val)
        return res
    
    
    def __setitem__(self, key, value):
        return self.arr.__setitem__(key, value)
    
    
    def __str__(self):
        return self.arr.__str__()
    
    
    def __repr__(self):
        return self.arr.__repr__()


    def copy(self):
        return Array(self.arr.copy())
    
    
    def __getattr__(self, attr):
        return self.arr.__getattribute__(attr)

    @staticmethod
    def concat(*args):
        return Array(list(args))
