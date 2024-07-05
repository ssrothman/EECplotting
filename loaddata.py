import numpy as np

def loaddata(dd):
    isdata = dd['isData']

    with open(dd['vals'], 'rb') as f:
        vals = np.load(f)

    with open(dd['covs'], 'rb') as f:
        covs = np.load(f)

    if dd['statsplit'] == 'none':
        pass
    elif dd['statsplit'] == 'sum':
        vals = np.sum(vals, axis=0)
        covs = np.sum(covs, axis=0)
    else:
        vals = vals[int(dd['statsplit'])]
        covs = covs[int(dd['statsplit'])]

    print(dd['vals'])
    print(vals.shape)
    print(covs.shape)

    return vals, covs, isdata
