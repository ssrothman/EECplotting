import numpy as np

def func(x, A, B, C):
    return A + B*x + C*x**2

def p0():
    return [0, 1, 0]

def get_text(popt):
    return '$y = (%+0.2g) + (%+0.2g) x + (%+0.2g) x^2$' % (popt[0], popt[1], popt[2])
