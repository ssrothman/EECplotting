import pickle

def efficiency(H):
    #shape MET, looseB, mediumB, tightB
    vals = H.values(flow=True)

    vals/=vals.sum()

    #print(vals.shape)
    #print(H)

    #B shape is (underflow, 0, 1, 2, overflow)
    #care about <2, so want vals[:3]

    loose  = vals[:,:,:,:].sum(axis=(1,2,3))
    medium = vals[:,:,:3,:].sum(axis=(1,2,3))
    tight  = vals[:,:,:,:3].sum(axis=(1,2,3))

    #for METcut in [41,51,61,71,81,101,126]:
    #    print(H.axes['MET'].edges[METcut])
    #    print("\tloose:", loose[:METcut].sum())
    #    print("\tmedium:", medium[:METcut].sum())
    #    print("\ttight:", tight[:METcut].sum())
    print(vals[:41,:3,:,:].sum()/vals[:41,:,:,:].sum())

paths = {
    'DYJetsToLL' : "/data/submit/srothman/EEC/Apr24_2024/DYJetsToLL/Kinematics/hists_file0to1825_tight_scanSyst.pkl", 
    'TTTo2L2Nu' : '/data/submit/srothman/EEC/Apr24_2024/TTTo2L2Nu/Kinematics/hists_file0to2924_tight_scanSyst.pkl',
    'WW' : '/data/submit/srothman/EEC/Apr24_2024/WW/Kinematics/hists_file0to319_tight.pkl',
    'ZZ' : '/data/submit/srothman/EEC/Apr24_2024/ZZ/Kinematics/hists_file0to71_tight.pkl',
    'WZ' : '/data/submit/srothman/EEC/Apr24_2024/WZ/Kinematics/hists_file0to164_tight.pkl',
    'ST_tW_top' : '/data/submit/srothman/EEC/Apr24_2024/ST_tW_top/Kinematics/hists_file0to157_tight.pkl',
    'ST_tW_antitop' : '/data/submit/srothman/EEC/Apr24_2024/ST_tW_antitop/Kinematics/hists_file0to158_tight.pkl',
    'ST_t_top' : '/data/submit/srothman/EEC/Apr24_2024/ST_t_top_5f/Kinematics/hists_file0to3955_tight.pkl',
    'ST_t_antitop' : '/data/submit/srothman/EEC/Apr24_2024/ST_t_antitop_5f/Kinematics/hists_file0to1984_tight.pkl',
    'DATA_2018A' : '/data/submit/srothman/EEC/Apr24_2024/DATA_2018A/Kinematics/hists_file0to1102_tight.pkl',
    'DATA_2018B' : '/data/submit/srothman/EEC/Apr24_2024/DATA_2018B/Kinematics/hists_file0to527_tight.pkl',
    'DATA_2018C' : '/data/submit/srothman/EEC/Apr24_2024/DATA_2018C/Kinematics/hists_file0to525_tight.pkl',
    'DATA_2018D' : '/data/submit/srothman/EEC/Apr24_2024/DATA_2018D/Kinematics/hists_file0to2515_tight.pkl'
}

for key in paths:
    print(key)
    with open(paths[key], 'rb') as f:
       Hdict = pickle.load(f)

    efficiency(Hdict['nominal']['selvar'])
    print()
    print()
