import numpy as np
import hist
from RecursiveNamespace import RecursiveNamespace

def Hdict_mult(Hdict, factor):
    for key in Hdict.keys():
        if type(Hdict[key]) in [hist.Hist, int, float, np.float32, np.float64]:
            Hdict[key] *= factor
        elif type(Hdict[key]) is dict:
            Hdict_mult(Hdict[key], factor)
        elif type(Hdict[key]) in [list, tuple, RecursiveNamespace]:
            pass
        else:
            raise ValueError("Hdict_mult: unknown type %s"%type(Hdict[key]))
    return Hdict

