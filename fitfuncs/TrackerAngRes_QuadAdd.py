import numpy as np

def func(x, A, B):
    return np.sqrt(np.square(A) + np.square(B/x))

def p0():
    return [0, 0]

def get_text(popt):
    return '$\\sigma = %0.2g \\oplus %0.2g/x$' % (popt[0], popt[1])
