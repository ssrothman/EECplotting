import numpy as np

def func(x, A, B, C):
    return A*np.power(x, B) + C

def p0():
    return [1, 1, 0]

def get_text(popt):
    return '$y = (%+0.2g) x^{(%+0.2g)} + (%+0.2g)$' % (popt[0], popt[1], popt[2])
