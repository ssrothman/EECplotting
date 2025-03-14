import numpy as np

def func(x, A, B, C):
    return A + B/np.power(x, 1+C)

def p0():
    return [0, 0, 0]

def get_text(popt):
    return '$\\sigma = %0.2g + %0.2g/x^{%0.2g}$' % (popt[0], popt[1], 1+ popt[2])
