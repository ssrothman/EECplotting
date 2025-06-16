import numpy as np
import matplotlib.pyplot as plt
import scipy

def invpoly4(x, params):
    return 1/poly4(x, params)

def invpoly5(x, params):
    return 1/poly5(x, params)

def invpoly6(x, params):
    return 1/poly6(x, params)

def poly2(x, params):
    return params[0] + params[1]*x

def poly3(x, params):
    return params[0] + params[1]*x + params[2]*x*x

def poly4(x, params):
    return params[0] + params[1]*x + params[2]*x*x + params[3]*x*x*x

def poly5(x, params):
    return params[0] + params[1]*x + params[2]*x*x + params[3]*x*x*x + params[4]*x*x*x*x

def poly6(x, params):
    return params[0] + params[1]*x + params[2]*x*x + params[3]*x*x*x + \
            params[4]*x*x*x*x + params[5]*x*x*x*x*x

def ratio21(x, params):
    return poly2(x, params[:2])/(1 + x * params[2])

def ratio22(x, params):
    return poly2(x, params[:2])/(1 + x * poly2(x, params[2:]))

def ratio23(x, params):
    return poly2(x, params[:2])/(1 + x * poly3(x, params[3:]))

def ratio24(x, params):
    return poly2(x, params[:2])/(1 + x * poly4(x, params[2:]))

def ratio31(x, params):
    return poly3(x, params[:3])/(1 + x * params[3])

def ratio32(x, params):
    return poly3(x, params[:3])/(1 + x * poly2(x, params[3:]))

def ratio33(x, params):
    return poly3(x, params[:3])/(1 + x * poly3(x, params[3:]))

def ratio34(x, params):
    return poly3(x, params[:3])/(1 + x * poly4(x, params[3:]))

def ratio41(x, params):
    return poly4(x, params[:4])/(1 + x * params[4])

def ratio42(x, params):
    return poly4(x, params[:4])/(1 + x * poly2(x, params[4:]))

def ratio43(x, params):
    return poly4(x, params[:4])/(1 + x * poly3(x, params[4:]))

def ratio44(x, params):
    return poly4(x, params[:4])/(1 + x * poly4(x, params[4:]))

def loss(x, y, yerr, params, func):
    fwd = func(x, params)
    err = y-fwd
    return np.sum(np.square(err/yerr)[np.isfinite(1/yerr)])

def minimize(x, ratio, ratioerr, fwdfunc, nparam, **kwargs):
    params0 = np.ones(nparam)

    theloss = lambda params : loss(x, ratio, ratioerr, params, fwdfunc)

    res = scipy.optimize.minimize(theloss, params0, **kwargs)
    print(res)

    plt.errorbar(x, ratio, yerr=ratioerr, fmt='o')
    xfine = np.linspace(x[0], x[-1], 100)
    plt.plot(xfine, fwdfunc(xfine, res.x))
    plt.show()
    return res
