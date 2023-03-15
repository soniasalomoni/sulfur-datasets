#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
#sys.path.insert(0,'/home/gigli/librascal/build/rascal/')

from matplotlib import pylab as plt


import ase
from ase.io import read, write
from ase.build import make_supercell
from ase.geometry.analysis import Analysis

from ase.visualize import view
import numpy as np
# If installed -- not essential, though
#try:
from tqdm.notebook import tqdm
#except ImportError:
    #tqdm = (lambda i, **kwargs: i)

from time import time

import skcosmo
import rascal
import sklearn
from rascal.models import Kernel, train_gap_model, compute_KNM, KRR
from rascal.representations import SphericalInvariants
from rascal.utils import from_dict, to_dict, CURFilter, FPSFilter, dump_obj, load_obj, get_score

import pickle as pk




LiPSdataset = read('input.xyz',':',format='extxyz')
Ndataset = len(LiPSdataset)
np.random.seed(50)
index = np.random.permutation(Ndataset)
LiPSdataset = [LiPSdataset[i] for i in index]




energiesXTB = np.array([snap.get_potential_energy() for snap in LiPSdataset ])
forcesXTB = np.array([snap.get_forces() for snap in LiPSdataset ],dtype=object)

forcesXTB_fl = np.concatenate(forcesXTB).ravel()



species = set()
for frame in LiPSdataset:
    species.update(frame.get_atomic_numbers())
species = np.sort(np.array(list(species)))

N_el = np.zeros((len(LiPSdataset), len(species)), dtype=int)
Natoms = np.zeros((len(LiPSdataset), 1), dtype=int)
atom_numbers = []

for i, frame in enumerate(LiPSdataset):
    atom_numbers = frame.get_atomic_numbers()
    Natoms[i] = frame.get_global_number_of_atoms()
    for j, s in enumerate(species):
        for el in atom_numbers:
            if(s == el): 
                N_el[i, j] += 1


# In[8]:


#energies_coesXTB = np.loadtxt('isolated_atom/PBEsol/atomic_energies_PBEsol.dat')
#print(Ndataset,energies_coesXTB)
#energies_coesiveXTB = np.array([np.sum(N_el[i]*energies_coesXTB) for i in range(Ndataset)])
#
#energiesXTB_mencoes = energiesXTB - np.array([np.sum(N_el[i]*energies_coesXTB) for i in range(Ndataset)])


# In[9]:




# In[8]:



av_forces = np.mean(forcesXTB_fl)
sigma_forces = np.std(forcesXTB_fl)

av_energies = np.mean(energiesXTB)
sigma_energies = np.std(energiesXTB)

print("Shape of the full input vector: ", forcesXTB_fl.shape[0] + energiesXTB.shape[0])
print("Mean & STD energies/atom = ", av_energies, " ", sigma_energies, " eV")
print("Mean & STD forces = ", av_forces, " ", sigma_forces, " eV/Angstrom")


# The very large variances on atomic energies that one sees are due to the fact that the $\alpha$ phase has a different 
# stoichiometry w.r.t. to $\beta$ and $\gamma$. The loss of one Li atoms accounts for the difference.

# In[9]:


for f in tqdm(LiPSdataset):
    f.wrap(eps=1e-18)


# In[10]:


Ndataset = len(LiPSdataset)
print(Ndataset)


species = set()
for frame in LiPSdataset:
    species.update(frame.get_atomic_numbers())
species = np.sort(np.array(list(species)))

N_el = np.zeros((len(LiPSdataset), len(species)), dtype=int)
Natoms = np.zeros((len(LiPSdataset), 1), dtype=int)
atom_numbers = []

for i, frame in enumerate(LiPSdataset):
    atom_numbers = frame.get_atomic_numbers()
    Natoms[i] = frame.get_global_number_of_atoms()
    for j, s in enumerate(species):
        for el in atom_numbers:
            if(s == el): 
                N_el[i, j] += 1


# In[13]:


lamb = 1.0
y=energiesXTB
X=N_el
Xt=X.transpose()
print(X.shape, Xt.shape)
XtX = np.dot(Xt, X)
Ndataset = len(LiPSdataset)

XtX_lamb = XtX + lamb * np.identity(len(XtX[0,:]))
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX_lamb,Xty)
energy_baseline_lstsq = {int(species): beta[i] for i, species in enumerate(species)}


# In[14]:


# Atomic energy baseline
atom_energy_baseline = np.sum([energiesXTB[i]/(LiPSdataset[i].get_global_number_of_atoms()) for i in range(len(LiPSdataset))])/Ndataset
energy_baseline = {int(sp): atom_energy_baseline for sp in species}
energy_zero = {int(sp): 1e-10 for sp in species}


# In[15]:


print(energy_baseline)
print(energy_baseline_lstsq)
print(energy_zero)


# In[23]:


hypersbaseline = []


# In[16]:


# define the parameters of the spherical expansion
hypersbaseline = dict(soap_type="PowerSpectrum",
              interaction_cutoff=6.0, 
              max_radial=8,
              max_angular=6,
              gaussian_sigma_constant=0.3,
              gaussian_sigma_type="Constant",
              cutoff_function_type="RadialScaling",
              cutoff_smooth_width=0.5,
              cutoff_function_parameters=
                    dict(
                            rate=1,
                            scale=3.5,
                            exponent=4
                        ),
              radial_basis="GTO",
              normalize=True,
              optimization=
                    dict(
                            Spline=dict(
                               accuracy=1.0e-05
                            )
                        ),
              compute_gradients=False
              )

repr_bas = SphericalInvariants(**hypersbaseline)


# In[18]:


for f in LiPSdataset: 
    f.wrap(eps=1.0e-12)


# In[19]:


managers = []
start = time()
managers = repr_bas.transform(LiPSdataset)
feat_vector = repr_bas.transform(LiPSdataset).get_features(repr_bas)
print ("Execution: ", time()-start, "s")


# In[23]:


# select the features with FPS, again the whole training set
n_sparse_feat = 448
#n_sparse_feat = 600*4
feat_compressor = FPSFilter(repr_bas, n_sparse_feat, act_on='feature')
feat_sparse_parameters = feat_compressor.select_and_filter(managers)


# In[24]:


feat_sparse_parameters.keys()
hypersbaseline['coefficient_subselection'] = feat_sparse_parameters['coefficient_subselection']
repr_bas_fsparse = SphericalInvariants(**hypersbaseline)


# In[25]:


start = time()
managers_sparsefeat = repr_bas_fsparse.transform(tqdm(LiPSdataset))
print("Execution time: {} s".format(time() - start))






# In[19]:


with open('kernels_zeta2_4000sparsefeat.pk','rb') as fg:
    dicts = pk.load(fg)
    
repr_bas_fsparse = dicts['repr_bas_fsparse']
kernel_base = dicts['kernel_base']
KNM_base = dicts['KNM_base']
X_sparse_baseline = dicts['X_sparse_baseline']


# In[17]:


del dicts


# # Optimizing the regularizers 
# In[177]:


ntrain = Ndataset
ntraining = [Ndataset]
Nkvalidation = 1
ids_validation = []
baseline = []

l1 = [1.0e-13, 1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5]

l2 = [1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0]
l2 = [1.0e-9,1.0e-8, 1.0e-7, 1.0e-6,1.0e-5, 1.0e-4, 1.0e-3]
l2 = [1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8, 1.0e-7, 1.0e-6]

#for lambda1 in [1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0]:
#    for lambda2 in [1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0]:
for lambda1 in l1:
    for lambda2 in l2:
        for k in range(Nkvalidation):
            ids = list(range(Ndataset))
            np.random.seed(20+k)
            np.random.shuffle(ids)
            ids_validation.append(ids)
            train_ids = [int(i) for i in ids[:ntrain]]
            frames_train = [LiPSdataset[ii] for ii in train_ids]

            en_train = [energiesXTB[ii] for ii in train_ids]
            #en_train = [energiesXTB_mencoes[ii]for ii in train_ids]
            en_train = np.array(en_train)

            f_train = [forcesXTB[ii] for ii in train_ids]
            f_train = np.array(f_train, dtype='object')
            f_train_fl = np.concatenate(f_train).ravel()
            #f_test_fl = np.concatenate(f_test).ravel()
            for nn in ntraining: 
                KnM_energies = np.array([KNM_base[ii,:] for ii in train_ids[:nn]])
                KnM_gradients = []

                for ii in train_ids[:nn]:
                    ntotatoms_prec_ii=0
                    for i, f in enumerate(LiPSdataset[:ii]):
                        ntotatoms_prec_ii+=f.get_global_number_of_atoms()

                    ntotatoms_ii = LiPSdataset[ii].get_global_number_of_atoms()
                    Init = Ndataset + ntotatoms_prec_ii*3
                    for ll in range(Init, Init+3*ntotatoms_ii): 
                        KnM_gradients.append(KNM_base[ll, :])

                KnM_gradients = np.array(KnM_gradients)
                print(KnM_gradients.shape, KnM_energies.shape, en_train.shape, f_train_fl.shape, len(KnM_gradients[:, 0]))
                KnM = np.vstack([KnM_energies, KnM_gradients])
                baseline.append(train_gap_model(kernel_base, frames_train[:nn], KnM, X_sparse_baseline, en_train[:nn], energy_baseline, 
                                    grad_train=-f_train_fl[:len(KnM_gradients[:, 0])], 
                                    lambdas=[lambda1, lambda2], jitter=1e-5,solver='RKHS-QR'))


# In[103]:


# baseline_energies = np.zeros((Ndataset, len(ntraining), Nkvalidation), dtype='float')
# baseline_forces = np.zeros((len(forces_fl), len(ntraining), Nkvalidation), dtype='float')
ngrid = len(l1)*len(l2)
baseline_energies = np.zeros((Ndataset, Nkvalidation, ngrid), dtype='float')
baseline_forces = np.zeros((len(forcesXTB_fl), Nkvalidation, ngrid), dtype='float')


# In[104]:


print(baseline_energies.shape)


# In[105]:


i = 0
iforce = 0
for f in tqdm(LiPSdataset): 
    f.wrap(eps=1e-18)
    m = repr_bas_fsparse.transform(f)
    for ng in range(ngrid):
        #for k in range(Nkvalidation):
        baseline_energies[i, 0, ng] = baseline[ng].predict(m)
        forces_tmp = baseline[ng].predict_forces(m)
        forces_tmp_fl = np.concatenate(forces_tmp).ravel()
        for l in range(iforce, iforce+len(forces_tmp_fl)):
                baseline_forces[l, 0, ng] = forces_tmp_fl[l-iforce]
    iforce += len(forces_tmp_fl)
    i += 1  


# In[106]:


baseline_forces.shape


# In[107]:


baseline_forces[:,0,ng]


# In[108]:


len(energiesXTB),forcesXTB.shape,len(LiPSdataset)


# In[110]:

ngj1 = [1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0]
for ng in range(ngrid):
    ee = []
    eML =[]
    for i,(s,e) in enumerate(zip(LiPSdataset,energiesXTB)):
        ee.append(e/len(s))
        eML.append((baseline_energies[i,0,ng])/len(s))
    
    score = get_score(np.array(ee), np.array(eML))
    RMSE_energy = score['RMSE']
                      
    score = get_score(np.array(baseline_forces[:,0,ng]), np.array(forcesXTB_fl))
    RMSE_forces  = score['RMSE']
    print("NG = ", ng)
    print("coeff = ",l1[ng//len(l2)],l2[ng%len(l2)]) #ngj1[ng//len(ngj1)],ngj1[ng%len(ngj1)])
    print("RMSE energy = ", RMSE_energy*1000.0, "meV/atom")
    print("RMSE forces = ", RMSE_forces*1000.0, "meV/force component = ", RMSE_forces/(sigma_forces)*100.0, "%")
        


# In[111]:


energiesXTB_mencoes,baseline_energies[:,0,0]


# In[112]:


forcesXTB_fl,baseline_forces[:,0,-1]


# In[113]:


energiesXTB,baseline_energies[:,0,32]


# In[43]:
RMSE_energies = np.zeros((len(solvers),len(ntraining), Nkvalidation), dtype='float')
RMSE_forces = np.zeros((len(solvers),len(ntraining), Nkvalidation), dtype='float')


for ng in range(36):
#     print(Nbeta, Nbeta211, Ngamma)
#     print(Nbeta + Nbeta211 + Ngamma)

    score_beta = get_score(np.array(baseline_en_beta)-np.mean(baseline_en_beta), np.array(energiesXTB_beta)-np.mean(energiesXTB_beta))
#     print(score_beta)
    RMSE_beta = score_beta['RMSE']

    score_gamma = get_score(np.array(baseline_en_gamma)-np.mean(baseline_en_gamma), np.array(energiesXTB_gamma)-np.mean(energiesXTB_gamma))
#     print(score_gamma)
    RMSE_gamma = score_gamma['RMSE']

    score_forces = get_score(np.array(baseline_forces[:,0,ng]), np.array(forcesXTB_fl))
#     print(score_gamma)
    RMSE_forces = score_forces['RMSE']
    print("NG = ", ng)
    print("RMSE beta = ", RMSE_beta*1000.0/(32.0), "meV/atom")
    print("RMSE gamma = ", RMSE_gamma*1000.0/(16.0), "meV/atom")
    print("RMSE forces = ", RMSE_forces*1000.0, "meV/force component = ", RMSE_forces/(sigma_forces)*100.0, "%")

