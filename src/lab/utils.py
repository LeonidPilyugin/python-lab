import numpy as np
import scipy as sc
import pint
import math
from uncertainties import ufloat
from .unitdict import udict
from .unit import unit

def uf(mean, std, units):
    return ufloat(mean, std) * unit(units)


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

def texify_nominal(val, err=None):
    if err is None or err == 0.0:
        return str(val).replace('.', '{,}')
    else:
        # get order
        val_order = math.floor(math.log10(abs(val)))
        err_order = math.floor(math.log10(abs(err)))

        # get first error digit
        first_err_digit = int(str(int(round((abs(err) / 10 ** err_order), 0)))[0])

        _first_err_digit = int((abs(err) / 10 ** err_order))

        if len(str(int(round((abs(err) / 10 ** err_order), 0)))) > 1:
            err_order += 1

        val_digits = val_order - err_order + 1
        err_digits = 1

        if first_err_digit == 1:
            err_digits += 1
            val_digits += 1

        if abs(err_order) > 2:
            val /= 10 ** err_order
            err /= 10 ** err_order
            val = round(val, val_digits - math.floor(math.log10(abs(val))) - 1)
            err = round(err, err_digits - math.floor(math.log10(abs(err))) - 1)
            if _first_err_digit != 1:
                err = round(err, 0)
            try:
                if val_digits - math.floor(math.log10(abs(val))) - 1 < 1:
                    val = int(val)
                    err = int(err)
            except Exception:
                val = 0
                err = int(err)
            return f"\\left({str(val).replace('.', '{,}')} \\pm {str(err).replace('.', '{,}')}\\right)\\cdot 10^{{{err_order}}}"
        else:
            val = round(val, val_digits - val_order - 1)
            err = round(err, err_digits - err_order - 1)

            if val_digits - val_order - 1 < 1:
                val = int(val)
                err = int(err)

            if err_digits == 2:
                if not any([str(i) in str(err).replace("1", "", 1) for i in range(1, 10)]):
                    if err_order < 0:
                        err = str(err) + "0"
                        val = str(val)
                        while len(val.split(".")[-1]) < abs(err_order) + 1:
                            val += "0"
                    if err_order == 0:
                        err = str(err).split(".")[0] + "{,}0"
                        val = str(val)
                        if not "." in val:
                            val = val + "{,}0"

            if err_order < 0:
                val = str(val)
                while len(val.split(".")[-1]) < abs(err_order):
                            val += "0"

            return f"{str(val).replace('.', '{,}')} \\pm {str(err).replace('.', '{,}')}"


def texify(value, dimension=True):
    if isinstance(value, pint.Unit):
        assert dimension
        return f"{value:~rutex}"
    elif isinstance(value, pint.Quantity):
        v = value.magnitude
        n = v if not hasattr(v, "n") else v.n
        s = None if not hasattr(v, "s") else v.s

        res = texify_nominal(n, s)

        if dimension:
            res += "\\;" + f"{value.units:~rutex}"

        return res
    else:
        return f"{str(value).replace('.', '{,}')}"

def totex(x, file=None, **kwargs):
    if hasattr(x, "texify"):
        string = x.texify(**kwargs)
        if file is None:
            print(string)
        else:
            with open(file, "w") as f:
                f.write(string)
                f.write("\n")
    else:
        print(texify(x))
