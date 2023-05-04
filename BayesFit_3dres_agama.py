#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paula G
@descrption: BayesFit for 3d resolution (with Agama vdf)
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
import agama
from multiprocessing import Pool
from scipy.special import gamma as gamma_fct
import pickle
from scipy.optimize import minimize


# Set units for AGAMA
agama.setUnits(mass=1,length=1,velocity=1)

#%%

# Global parameters

# Read in data files
DATAFILE = 'mock_3dres_nstars_100seed_1234.txt'
data_ = np.loadtxt(DATAFILE)

NSTARS = len(data_)

x  = data_[:,0]
y  = data_[:,1]
vz = data_[:,2]
#mass = data_[:,6]
data = np.column_stack((x,y,vz))
data_proj = np.column_stack((x,y,x*0,x*0,vz,x*np.inf,x*np.inf,x*0)) #prepare data file for marginalisation

VERBOSE     = True 
AUTOCORR    = False # autocorrelation analysis?
NSTEPS      = 20000 # number steps in mcmc
NPARAM      = 13
NWALKERS    = 26
MP          = True
CLUSTER     = True
RESTART     = True
BACKEND     = True
BACKENDFILE = "Backend_mock_3d_nstars_"+str(NSTARS)+".h5"
#%%
print('*********** BayesFit to Auriga simulations  ******************')
print('Number of walkers: '+str(NWALKERS))
print('Number of steps in emcee: '+str(NSTEPS))
print('Number of stars in data file: '+str(NSTARS))
print('Data file name: '+ DATAFILE)
print('Backend file name: '+ BACKENDFILE)
print('**************************************************************')

# =============================================================================
#%% Define prior, likelihood, and posterior functions

def _posterior(param):
    """
    

    Parameters
    ----------
    param : TYPE [numpy.array]
        DESCRIPTION: Contains the parameters of the model. slopein, slopeout, J_0, coefJrIn (hr) -  characterize
        the distribution function (df) of the stellar halo. 
        NFW dark matter halo potential parameters: rho_0 (scale density of the NFW profile), Rs (scale radius), q (flattening, z-axis ratio).
        of the dark matter halo.

    Returns: logL
    -------
    TYPE [scalar]
        DESCRIPTION: Returns the value of the posterior for the given model.

    """
    #slopein,slopeout,J_0 = param
    slopein,slopeout,J_0, coefJrIn, coefJzIn,coefJrOut,coefJzOut, rotFrac, rho0, Rs, q, mass_bulge,mass_disk = param
    
    # Prior
    if (slopein<0) | (slopein>3.) | (slopeout<0) | (slopeout<3.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf

    if  (J_0<=0.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf
    
    if  (coefJrIn<0.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf
    
    if  (coefJzIn<0.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf
    
    if (3-coefJzIn-coefJrIn<0.):
        return -np.inf
    
    if  (coefJrOut<0.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf
    
    if  (coefJzOut<0.):
        if VERBOSE ==True:
            print ('**from prior -inf')
        return -np.inf
    if (3-coefJzOut-coefJrOut<0.):
        return -np.inf
    
    if (rotFrac<-1) | (rotFrac>1):
        return -np.inf
    
    if (rho0<0.):
        return -np.inf
    if (Rs<0.):
        return -np.inf
    if (q<=0.) | (q>1.):
        return -np.inf
    if (mass_bulge<=0.):
        return -np.inf
    if (mass_disk<=0.):
        return -np.inf
# =============================================================================
#     if (beta<=2.):
#         return -np.inf
#     if(gamma>=3):
#         return -np.inf
# =============================================================================

    
    # Define potential
    
    # Potential for the halo (NFW)
# =============================================================================
#     alpha = 1. # ?
#     beta  = 3.
#     gamma = 1.
# =============================================================================
    r_cut = 1000000.

    dmHaloNFW_param = dict(type='Spheroid',
                    densityNorm=rho0,
                    gamma=1.,
                    beta=3.,
                    alpha=1.,
                    scaleRadius = Rs,
                    outercutoffradius=r_cut,
                    axisRatioZ = q)

    pot_halo = agama.Potential(dmHaloNFW_param)
    
    # Potential for the disk
    #k_disk = gamma_fct(2.0*1.2)/gamma_fct(3.0*1.2)  
    pot_disk = agama.Potential(type='Disk',
                               #mass=5.6*(10**10),
                               mass = mass_disk,
                               scaleRadius=2.57,
                               #scaleRadius = RDISK,
                               scaleHeight=0.4,
                               n=1.2)


    # Potential for the bulge
    pot_bulge = agama.Potential(type='Sersic',
                        #mass=3.1*(10**10),
                        mass = mass_bulge,
                        scaleRadius = 1.155,
                        #scaleRadius = RBULGE,
                        sersicIndex=2.7,
                        axisRatioZ=0.72)
    # Total potential
    pot      = agama.Potential(pot_bulge,pot_disk,pot_halo) # total potential of the galaxy

    # Distribution function for the stellar halo
    df = agama.DistributionFunction(type ='DoublePowerLaw',
                                    norm=1,
                                    slopeIn=slopein,
                                    slopeOut = slopeout,
                                    J0=J_0,
                                    coefJrIn = coefJrIn,
                                    coefJzIn=coefJzIn,
                                    coefJrOut=coefJrOut,
                                    coefJzOut=coefJzOut,
                                    rotFrac = rotFrac,
                                    Jcutoff=12000.) 
    
    norm = df.totalMass()
    
    # Calculate likelihood (+ marginalise over unknown coordinates (z,vx,vy))
    try:
        gm = agama.GalaxyModel(pot,df)
    except (RuntimeError):
        print('RuntimeError from creating GalaxyModel')
        return -np.inf
    try:
        DF = gm.projectedDF(data_proj)
    except (RuntimeError):
        print('RuntimeError from creating calculating projected df')
        return -np.inf
    
    logL = np.sum(np.log(DF/norm)) # assuming all particles have equal mass
    
    
    #logL = np.sum(np.log((df(act_f)/norm)*mass)) # total likelihood (of all the stars in the catalog)
    
    if np.isnan(logL):
        print('nan logL')
        logL = -np.inf
    return (logL)


#%% Apply gradient optimization

initial_guess = np.array([1.8,5.,7000., 0.8,1.3,0.7,0.85,0.3,1.4*(10**7.),16.,0.7,2.8*(10**10),5.1*(10**10)])
param_opt = initial_guess
# =============================================================================
# opt = minimize(_minus_likelihood,initial_guess,method='Nelder-Mead')
# param_opt = opt.x
# print("Parameters that minimize -logL: "+ str(param_opt))
# =============================================================================
#%% emcee 
def main(nsteps,nwalkers,param_opt,mp,backendfile):
    
    # Set random number seed
    
    # Number of dimensions
    ndim = len(param_opt)
    
    #pos = np.random.randn(nwalkers,ndim) # starting point(s) of the walkers
    np.random.seed(1234)

    #pos = param_opt + 1e-4 * np.random.randn(nwalkers, ndim) # starting point(s) of the walkers

    
    print("...using "+str(nwalkers)+" walkers...")
    
    # Define backend 

    

    with Pool() as pool:
        print('...in parallel mode...')
        
        if (RESTART):
            backend = emcee.backends.HDFBackend(backendfile) 
            sampler = emcee.EnsembleSampler(nwalkers,ndim, _posterior,pool=pool,backend=backend)
            pos = sampler.get_last_sample()
                     
        else:
            pos = param_opt + 1e-2 * np.random.randn(nwalkers, ndim) # starting point(s) of the walkers
            backend = emcee.backends.HDFBackend(backendfile)
            backend.reset(nwalkers,ndim)
            sampler = emcee.EnsembleSampler(nwalkers,ndim, _posterior,pool=pool,backend=backend)
        
        sampler.run_mcmc(pos, nsteps,progress=True,store=True)
#%%
if __name__ == "__main__":
    print('Starting emcee run...')
    start = time.time()
    main(NSTEPS,NWALKERS,param_opt,MP,BACKENDFILE)
    stop = time.time()
    multi_time = stop-start
    print(" ")
    print("emcee took "+str(np.round(multi_time,2))+" s.")
    print('Done!')
    
#%%

# =============================================================================
# n = 10
# slopein_ = np.linspace(0.00001,2.9999,n)
# #slopeout_ = np.linspace(3.0001,10,n) 
# #J0_ =np.linspace(5000,10000,n)
# post = []
# 
# start1 = time.time()
# for i in range(n):
#     par_ = np.array([slopein_[i],5.5,8000, 0.75,1.7,0.88,1.1,0.5,1.1*(10**7.),17.,0.8,3.1*(10**10),5.6*(10**10)])
#     post.append(_posterior(par_))
# end1 = time.time()
# 
# post = np.array(post)
# t = end1-start1
# meant= t/n
# #%%
# 
# with plt.xkcd():
#     # This figure will be in XKCD-style
#     fig = plt.figure()
#     plt.plot(slopein_,post,c='black',label=r'proj DF,$\bar{t}$ =20.20s/it')
#     plt.xlabel(r'$\beta_{in}$')
#     plt.ylabel('posterior')
#     plt.legend()
#     plt.savefig('projDF_slopein.png',format='png',bbox_inches='tight')
# =============================================================================









