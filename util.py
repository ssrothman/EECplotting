import numpy as np
import hist

def is_integral(dtype):
    return dtype in [np.bool_, np.int16, np.int32, np.int64, 
                     np.uint16, np.uint32, np.uint64]

def make_ax(dtype, minval, maxval, nbins, logx):
    if type(minval) in [list, tuple]:
        theax = hist.axis.Variable(
            minval, 
        )
    elif is_integral(dtype):
        theax = hist.axis.Integer(minval, maxval+1)
        if logx:
            raise ValueError("logx is not supported for integer axes")
    else:
        if logx and minval <= 0:
            minval = 1

        theax = hist.axis.Regular(
                nbins, minval, maxval,
                transform=hist.axis.transform.log if logx else None
        )

    return theax

def histogram_ratio(H1, H2, flow=True):
    num = H1.values(flow=flow)
    denom = H2.values(flow=flow)

    numerr = np.sqrt(H1.variances(flow=flow))
    denomerr = np.sqrt(H2.variances(flow=flow))

    ratio = num/denom
    ratioerr = ratio * np.sqrt(np.square(numerr/num) + np.square(denomerr/denom))

    return ratio, ratioerr
