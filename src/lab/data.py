import pandas as pd
import numpy as np
from uncertainties import ufloat
from .unit import unit
from .unitdict import udict
from . import utils
from . import arr

class Data:
    def __init__(self, path, **kwargs):
        self.load(path, **kwargs)
        
        
    def load(self, path, preprocess = None, dimension = None, begin = lambda x: x):
        # read
        df = begin(pd.read_csv(path))
        
        # preprocessing
        if preprocess is None: preprocess = {}
        for column, function in preprocess.items():
            df[column] = df[column].apply(function)
        
        # dimension
        if dimension is None:
            dimension = df.iloc[0].fillna("dimensionless")
            df = df.drop(index=[0])
        else:
            assert len(dimension) == len(df.columns)
            _dimension = dimension
            dimension = {}
            for i in range(len(_dimension)):
                dimension[df.columns[i]] = _dimension[i]
        
        # get nominal and error columns
        columns = {}
        for c in df.columns:
            if c.endswith("err"):
                assert c.split(".")[0] in columns.keys()
                columns[c.split(".")[0]] = c
            else:
                columns[c] = None
                
        # create measurements
        df2 = pd.DataFrame()
        for k, ec in columns.items():
            if ec is None:
                res = []
                for i in range(len(df[k])):
                    res.append(df[k].iloc[i] * unit(dimension[k]))
                
                df2[k] = np.array(res, dtype=object)
                continue
            
            prev = 0.0
            for i in range(len(df[ec])):
                if pd.isna(df[ec].iloc[i]):
                    df[ec].iloc[i] = prev
                else:
                    prev = df[ec].iloc[i]

            def handle_percent(x):
                if "%" in x:
                    x = float(df[k].iloc[handle_percent.counter]) * float(x.strip("%")) / 100
                handle_percent.counter += 1
                return x

            handle_percent.counter = 0

            df[ec] = df[ec].apply(handle_percent)

            df2[k] = utils.create_measure(df[k].to_numpy(dtype="float") * unit(dimension[k]), df[ec].to_numpy(dtype="float") * unit(dimension[ec]))
        
        
        self.primary_measurements = df2.copy()
        
        # process repeated measurments
        if "number" in df.columns:
            # process repeated measurments
            df = df2
            l = []
            for d in df.groupby("number"):
                f = d[1]
                
                if len(f) == 1:
                    l.append(f.iloc[0].values.tolist())
                    continue
                
                arr = [d[0]]
                for c in f.columns:
                    if hasattr(f[c].iloc[0], "n"):
                        temp = utils.normalize(f[c].to_numpy(dtype=object))
                        units = temp[0].units
                        err = ufloat(0, max(utils._std(temp))) * units
                        temp = utils._nominal(temp)
                        mean = ufloat(temp.mean(), 0) * units
                        std = ufloat(0, temp.std() / np.sqrt(len(temp))) * units
                        arr.append(mean + std + err)
                    else:
                        temp = f[c].to_numpy(dtype=object)
                        mean = temp.mean()
                        std = temp.std()
                        
                        assert std == 0.0
                        
                        f[c] = mean
                    
                l.append(arr)
                    
            self.df = pd.DataFrame(l, dtype=object, columns=columns)
        else:
            self.df = df2.copy()
        
        
    def __getitem__(self, key):
        return arr.Array(self.df[key].to_numpy(dtype=object))
    
    
    def __setitem__(self, key, value):
        self.df[key] = value.arr
    
        
    def texify(self, primary=False, columns=None):
        df = self.primary_measurements if primary else self.df
        
        result = "\\begin{tabular}{|" + "|".join(["l"] * len(df.columns)) + "|}\n"
        
        temp = []
        
        for c in df.columns:
            cname = c if columns is None else columns[c]
            s = f"${cname}" if cname != "number" else "№"
            
            if hasattr(df[c].iloc[0], "dimensionless") and not df[c].iloc[0].dimensionless:
                s += f",\\;{utils.texify(df[c].iloc[0].units)}"
            
            if cname != "number": s += "$"
                
            temp.append(s)
            
        result += "\\hline\n" + " & ".join(temp) + "\\\\\\hline\n"
        
        for _, row in df.iterrows():
            result += " & ".join(["$" + utils.texify(r, False) + "$" for r in row]) + "\\\\\\hline\n"
        
        result += "\\end{tabular}"
        return result
    
    
    def __getattr__(self, attr):
        return self.df.__getattribute__(attr)
