import numpy as np
import eigenpy as eigen

cov = np.load("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL/EECproj/hists_file0to6276_tight_statsplit2_manualcov___covreco.npy")
recopure = np.load("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL/EECproj/hists_file0to6276_tight_statsplit2_manualcov___recopure.npy")
genpure = np.load("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL/EECproj/hists_file0to6276_tight_statsplit2_manualcov___genpure.npy")
transfer_in = np.load("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL/EECproj/hists_file0to6276_tight_statsplit2_manualcov___transfer.npy")

cov = np.sum(cov, axis=0)
recopure = np.sum(recopure, axis=0)
genpure = np.sum(genpure, axis=0)
transfer_in = np.sum(transfer_in, axis=0)

sumtransfer_in = np.sum(transfer_in, axis=(4,5,6))
sumtransfer_in_2 = np.sum(transfer_in, axis=(1,2,3))

print(np.allclose(sumtransfer_in, recopure))

transfer = np.zeros((*recopure.shape, *recopure.shape))
for a in range(transfer.shape[0]):
    for b in range(transfer.shape[1]):
        for c in range(transfer.shape[2]):
            for d in range(transfer.shape[3]):
                for i in range(transfer.shape[4]):
                    for j in range(transfer.shape[5]):
                        for k in range(transfer.shape[6]):
                            for l in range(transfer.shape[7]):
                                if a == i:
                                    transfer[a,b,c,d,i,j,k,l] = transfer_in[a,b,c,d,j,k,l]
                                else:
                                    transfer[a,b,c,d,i,j,k,l] = 0

#transfer = np.einsum('abcdijk->aijkabcd', transfer)

print(transfer.shape)
print(cov.shape)
print(recopure.shape)
print(genpure.shape)

sumtransfer = np.sum(transfer, axis=(4,5,6,7))
print(np.allclose(sumtransfer, recopure))

invgen = 1/genpure
invgen[genpure==0] = 0

transfer = np.einsum('abcdijkl,ijkl->abcdijkl', transfer, invgen)

N = np.prod(recopure.shape)

transfer = transfer.reshape((N,N))
cov = cov.reshape((N,N))
recopure = recopure.reshape((N,))
genpure = genpure.reshape((N,))

forward = transfer @ genpure

print(np.allclose(forward, recopure))


def eigenunfold(cov, recopure):
    llt = eigen.LLT(cov)
    L = llt.matrixL()

    codL = eigen.CompleteOrthogonalDecomposition(L)
    Lreco = codL.solve(recopure)
    Ltransfer = codL.solve(transfer)

    codLtransfer = eigen.CompleteOrthogonalDecomposition(Ltransfer)

    unfolded = codLtransfer.solve(Lreco)

    return unfolded

#unfolded_eigen = eigenunfold(cov, recopure)
#print(np.allclose(unfolded_eigen, genpure))

from torchunfold import *

#recopure = recopure[:10]

#testcov = torch.Tensor(np.eye(len(recopure)))
cov = cov + 1e-6*np.eye(len(recopure))
codcov = eigen.CompleteOrthogonalDecomposition(cov)
invcov = codcov.solve(np.eye(len(recopure)))

testcov = torch.Tensor(invcov).cuda()
#testtransfer = torch.Tensor(np.eye(len(recopure)))
testtransfer = torch.Tensor(transfer).cuda()
testrecopure = torch.Tensor(recopure).cuda()

#print(testcov.shape)
#print(testtransfer.shape)
#print(testrecopure.shape)


mod = TorchUnfolding(testcov, testtransfer, testrecopure)
opt = torch.optim.AdamW(mod.parameters(), lr=1e-3)

from tqdm import tqdm
t = tqdm(range(1000))

mod = mod.cuda()
testcov = testcov.cuda()
testtransfer = testtransfer.cuda()
testrecopure = testrecopure.cuda()

for epoch in t:
    loss = mod.forward(testrecopure)
    loss.backward()
    opt.step()
    opt.zero_grad()
    t.set_description(f"Loss: %g" % loss.item())

print(mod.unfolded.data.cpu().numpy()[:10])
print(genpure[:10])
