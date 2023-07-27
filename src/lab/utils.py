import numpy as np
import scipy as sc
import pint
import math
from uncertainties import ufloat
from .unitdict import udict

_nominal = np.vectorize(lambda x: x.n)
_units = np.vectorize(lambda x: x.units)
_std = np.vectorize(lambda x: x.s)
magnitude = np.vectorize(lambda x: x.magnitude)


def nominal(x):
    if isinstance(x, np.ndarray):
        return _nominal(x) * _units(x)
    else:
        return x.n * x.units
    

def std(x):
    if isinstance(x, np.ndarray):
        return _std(x) * _units(x)
    else:
        return x.s * x.units
    

def convert(array, units):
    res = []
    
    for a in array:
        res.append(a.to(units))
        
    return np.array(res, dtype=object)


def normalize(array):
    units = array[0].units
    res = []
    
    for a in array:
        res.append(a.to(units))
    
    return np.array(res, dtype=object)


def create_measure(n, s):
    assert len(s) == len(n)
    
    res = []
    n = normalize(n)
    s = convert(s, n[0].units)
    
    for i in range(len(s)):
        res.append(ufloat(n[i].magnitude, s[i].magnitude) * n[0].units)
        
    return np.array(res, dtype=object)


def student(n, confidence=0.95):
    return sc.stats.t.ppf((1 + confidence) / 2, n - 1)
        
        
@pint.register_unit_format("rutex")
def format_unit(unit, registry, **options):
    
    numerator = []
    denominator = []
    for u, p in unit.items():
        (numerator if p > 0 else denominator).append((u, abs(p)))
    
    res = "\\cdot".join(["\\text{" + udict[n[0]] + ("" if n[1] == 1 else "}^{" + str(n[1])) + "}" for n in numerator])
    
    if len(denominator) > 0 and len(numerator) > 0: res += " / "
    if len(numerator) == 0:
        for i in range(len(denominator)):
            denominator[i] = (denominator[i][0], -denominator[i][1])
         
    res += "\\cdot".join(["\\text{" + udict[n[0]] + ("" if n[1] == 1 else "}^{" + str(n[1])) + "}" for n in denominator])
    
    return res
        
        
def texify(value, dimension=True):
    if isinstance(value, pint.Unit):
        assert dimension
        return f"{value:~rutex}"
    elif isinstance(value, pint.Quantity):
        v = value.magnitude
        n = v if not hasattr(v, "n") else v.n
        e = math.floor(math.log10(n))
        if e < -1 or e > 3:
            res = "{:e}".format(v)
        else:
            res = str(v)
        
        if "e" in res:
            man, exp = res.split("e")
            exp = exp.replace("+0", "")
            exp = exp.replace("-0", "-")
            exp = exp.replace("+", "")
            res = man + "\\cdot 10^{" + exp + "}"
            
        res = res.replace("(", "\\left(")
        res = res.replace(")", "\\right)")
        res = res.replace("+/-", "\\pm")
        res = res.replace(".", "{,}")
            
        if dimension:
            res += "\\;" + f"{value.units:~rutex}"
            
        return res
    else:
        return f"{str(value).replace('.', '{,}')}"


def totex(x, **kwargs):
    if hasattr(x, "texify"):
        print(x.texify(**kwargs))
    else:
        print(texify(x))
