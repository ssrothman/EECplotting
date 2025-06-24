import argparse

parser = argparse.ArgumentParser()

parser.add_argument("Runtag")
parser.add_argument("Skimmer")
parser.add_argument("Objsyst")
parser.add_argument("Wtsyst")
parser.add_argument("what")

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--reweight', type=str, default='Pythia_Zkinweight')
parser.add_argument('--max_nboot', type=int, default=-1)

parser.add_argument('--outputtag', type=str, default='Pythia_HTsum')

args = parser.parse_args()

import json

with open("config/datasets.json", 'rb') as f:
    datasets_config = json.load(f)

dsets = datasets_config['StacksMC']['HT']['dsets']
xsecs = [datasets_config['DatasetsMC'][dset]['xsec'] for dset in dsets]

import datasets

H = datasets.get_pickled_histogram_sum(
        dsets, xsecs, args.Runtag, args.Skimmer,
        args.Objsyst, args.Wtsyst, args.what,
        statN=args.statN, statK=args.statK,
        shuffle_boots=False,
        boot_per_file=args.boot_per_file,
        reweight=args.reweight,
        max_nboot=args.max_nboot)

import os
outfile = '%s_%s_%s'%(args.what, args.Objsyst, args.Wtsyst)
nboot = H.axes['bootstrap'].size - 1
if nboot > 0:
    outfile += '_boot%d' % nboot
if args.statN > 0:
    outfile += '_%dstat%d' % (args.statN, args.statK)
if args.reweight is not None:
    outfile += '_%s' % args.reweight
outfile += "_HTSUM.pkl"

output_path = os.path.join(datasets.basedir, args.Runtag, 
                           args.outputtag, args.Skimmer, 
                           outfile)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    import pickle
    pickle.dump(H, f)
