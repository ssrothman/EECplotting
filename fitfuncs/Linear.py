import numpy as np

def func(x, A, B):
    return A + B*x

def p0():
    return [0, 1]

def get_text(popt):
    return '$y = %0.2g + %0.2g x$' % (popt[0], popt[1])
