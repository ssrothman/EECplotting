import numpy as np

def func(x, A):
    return A*x

def p0():
    return [1]

def get_text(popt):
    return '$y = (%+0.2g) x$' % popt[0]
