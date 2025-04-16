import numpy as np

def func(x, A, B):
    return A*np.power(x, B)

def p0():
    return [1, 1]

def get_text(popt):
    return '$y = %0.2g x^{%0.2g}$' % (popt[0], popt[1])
