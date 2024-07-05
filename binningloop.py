import itertools
from util import ax_extent, config

def binningloop(command):
    bases = []
    for fix in command['fixes']:
        bases.append({})
        for ax in fix:
            bases[-1][ax] = fix[ax]

    names_to_loop = command['loopover']
    extents_to_loop = [ax_extent(name) for name in names_to_loop]
    ranges = [range(extent) for extent in extents_to_loop]
    product = itertools.product(*ranges)

    for p in product:
        binning_l = [base.copy() for base in bases]
        for i, name in enumerate(names_to_loop):
            for j in range(len(binning_l)):
                binning_l[j][name] = p[i]

        yield binning_l

