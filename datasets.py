import pyarrow.dataset as ds
import pickle
import os.path

basedir = '/ceph/submit/data/user/s/srothman/EEC/'
unf_basedir = '/home/submit/srothman/work/EEC/EECunfold/data'

def get_dataset(runtag, tag, skimmer, objsyst, whichobj):
    thepath = os.path.join(basedir, runtag, tag, skimmer)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if not (subpath.is_dir()):
            continue
        options.append(subpath.name)

    if (len(options) == 1):
        print()
        print("Only one option found: %s" % options[0])
        print("No user input needed :D")
        print()
        thepath = os.path.join(thepath, options[0])
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thepath = os.path.join(thepath, options[choice])
        print()

    thepath = os.path.join(thepath, objsyst, whichobj)
    print("The path is: %s" % thepath)

    return ds.dataset(thepath, format="parquet")

def get_counts(runtag, tag):
    thepath = os.path.join(basedir, runtag, tag, 'Count')

    print(thepath)
    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        options.append(subpath.name)

    if (len(options) == 1):
        print()
        print("Only one option found: %s" % options[0])
        print("No user input needed :D")
        print()
        thepath = os.path.join(thepath, options[0])
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thepath = os.path.join(thepath, options[choice])
        print()

    print("The path is: %s" % thepath)

    with open(thepath, 'rb') as f:
        return pickle.load(f)['num_evt']

def get_unfolded_histogram(name):
    with open(os.path.join(unf_basedir, name+'.pkl'), 'rb') as f:
        return pickle.load(f)

def get_pickled_histogram(runtag, tag, skimmer, objsyst, wtsyst, whichobj):
    thepath = os.path.join(basedir, runtag, tag, skimmer)

    subpaths = os.scandir(thepath)
    options = []
    for subpath in subpaths:
        if not (subpath.is_dir()):
            continue
        options.append(subpath.name)

    if (len(options) == 1):
        print()
        print("Only one option found: %s" % options[0])
        print("No user input needed :D")
        print()
        thepath = os.path.join(thepath, options[0])
    else:
        print()
        print("Multiple options found:")
        for i, option in enumerate(options):
            print(f"{i}: {option}")
        print()
        choice = int(input("Please select a number: "))
        thepath = os.path.join(thepath, options[choice])
        print()

    thepath = os.path.join(thepath, objsyst, '%s_%s_%s.pkl'%(whichobj, objsyst, wtsyst))
    print("The path is: %s" % thepath)

    with open(thepath, 'rb') as f:
        return pickle.load(f)
