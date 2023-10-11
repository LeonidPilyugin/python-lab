import matplotlib.pyplot as plt
import pint
from . import utils
from . import arr
from itertools import combinations
from .unit import unit
from .unitdict import udict, mpludict

plt.style.use("https://raw.githubusercontent.com/LeonidPilyugin/mpl-style/main/simple.mplstyle")

@pint.register_unit_format("plot")
def format_unit(unit, registry, **options):
    def part(x):
        part.cdot = "\u00B7"

        part.replacements = {
            "0": "\u2070",
            "1": "\u00b9",
            "2": "\u00b2",
            "3": "\u00b3",
            "4": "\u2074",
            "5": "\u2075",
            "6": "\u2076",
            "7": "\u2077",
            "8": "\u2078",
            "9": "\u2079",
        }

        num = []
        for n in x:
            temp = mpludict.get(n[0], udict[n[0]])

            if n[1] != 1:
                ee = str(n[1])
                for key, rep in part.replacements.items():
                    ee = ee.replace(key, rep)
                temp += ee
            num.append(temp)

        res = part.cdot.join(num)
        return res


    numerator = []
    denominator = []
    for u, p in unit.items():
        (numerator if p > 0 else denominator).append((u, abs(p)))

    res = part(numerator)
    if len(denominator) > 0:
        if len(numerator) == 0:
            res += "1"
        res += "/"
        res += part(denominator)

    return res


class Plot:
    def __init__(self, **kwargs):
        self.fig, self.ax = plt.subplots()
        self.set_x_quantity(kwargs.get("xq", None))
        self.set_y_quantity(kwargs.get("yq", None))
        self.set_xlabel(kwargs.get("xl", ""))
        self.set_ylabel(kwargs.get("yl", ""))
        self.set_title(kwargs.get("title", ""))
        
        
    def set_x_quantity(self, q):
        self._qx = q


    def set_y_quantity(self, q):
        self._qy = q
        

    def set_quantity(self, qx, qy):
        self.set_x_quantity(qx)
        self.set_y_quantity(qy)
    
        
    def set_xlabel(self, label):
        self._xlabel = label
        self._update_xlabel()
    
    
    def set_ylabel(self, label):
        self._ylabel = label
        self._update_ylabel()
        
        
    def set_title(self, title):
        self._title = title
        self._update_title()
        
        
    def _update_xlabel(self):
        xl = self._xlabel
        if not self._qx is None and not self._qx.dimensionless:
            xl += f", {self._qx:~plot}"
        self.ax.set_xlabel(xl)
        
        
    def _update_ylabel(self):
        yl = self._ylabel
        if not self._qy is None and not self._qy.dimensionless:
            yl += f", {self._qy:~plot}"
        self.ax.set_ylabel(yl)
        
    
    def _update_title(self):
        self.ax.set_title(self._title)
    
    
    def plot(self, x, y, **kwargs):
        x = arr.Array(x)
        y = arr.Array(y)

        if self._qx is None:
            self._qx = x.arr[0].units
            self._update_xlabel()
        if self._qy is None:
            self._qy = y.arr[0].units
            self._update_ylabel()

        # convert units
        x = utils.convert(x.arr, self._qx)
        y = utils.convert(y.arr, self._qy)
        
        # split x and y on nominal and error
        if hasattr(x[0], "s"):
            xerr = utils.magnitude(utils.std(x))
            x = utils.magnitude(utils.nominal(x))
        else:
            xerr = None
            x = utils.magnitude(x)
        if hasattr(y[0], "s"):
            yerr = utils.magnitude(utils.std(y))
            y = utils.magnitude(utils.nominal(y))
        else:
            yerr = None
            y = utils.magnitude(y)
        
        # plot
        self.ax.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)
        
    
    def line(self, k, b, **kwargs):
        if not isinstance(k, float):
            k = k.to(self._qy / self._qx).magnitude.n
        if not isinstance(b, float):
            b = b.to(self._qy).magnitude.n
        xi, xa = self.ax.get_xlim()
        self.ax.axline([xi, xi * k + b], [xa, xa * k + b], **kwargs)
        
    
    def errline(self, k, b, alpha=0.1, **kwargs):
        xi, xa = self.ax.get_xlim()
        
        k = k.to(self._qy / self._qx)
        b = b.to(self._qy)
        
        k1 = k.magnitude.n + k.magnitude.s
        k2 = k.magnitude.n - k.magnitude.s
        
        b1 = b.magnitude.n + b.magnitude.s
        b2 = b.magnitude.n - b.magnitude.s
        
        self.line(k, b, color="black", linestyle=(0, (5, 10)))
        
        for l1, l2 in combinations([(k1, b1), (k1, b2), (k2, b1), (k2, b2)], 2):
            _k1, _b1 = l1
            _k2, _b2 = l2
            
            self.ax.fill_between([xi, xa], [xi * _k1 + _b1, xa * _k1 + _b1], [xi * _k2 + _b2, xa * _k2 + _b2], alpha=alpha, **kwargs)


    def legend():
        self.ax.legend()
        
    
    def clear(self):
        self.ax.cla()
        
        
    def save(self, path, **kwargs):
        self.fig.savefig(path, **kwargs)
    
