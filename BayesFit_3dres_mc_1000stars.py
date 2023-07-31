#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paula G
@descrption: BayesFit for 3d resolution.

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
#DATAFILE = 'mock_3dres_nstars_100seed_1234.txt'
# =============================================================================
# DATAFILE = "mock_3dres_nstars_100_highJcutoff.txt"
# data_ = np.loadtxt(DATAFILE)
# 
# NSTARS = 100
# 
# x  = data_[:,0]
# y  = data_[:,1]
# vz = data_[:,2]
# #mass = data_[:,6]
# data = np.column_stack((x,y,vz))
# =============================================================================

DATAFILE_MC = 'data_mc_nstars_1000_nmc_1000.pkl'
with open(DATAFILE_MC,'rb') as f:
    data_mc = pickle.load(f)


NSTARS      = 1000
NMC         = 1000 # number of mc samples for marginalisation calculation
VERBOSE     = True 
AUTOCORR    = False # autocorrelation analysis?
NSTEPS      = 40000 # number steps in mcmc
NPARAM      = 13
NWALKERS    = 26
MP          = True
CLUSTER     = True
RESTART     = False
BACKEND     = True
BACKENDFILE = "Backend_3d_MC_nstars_"+str(NSTARS)+".h5"
#%%
print('*********** BayesFit to Auriga simulations  ******************')
print('Number of walkers: '+str(NWALKERS))
print('Number of steps in emcee: '+str(NSTEPS))
print('Number of stars in data file: '+str(NSTARS))
print('Data file name: '+ DATAFILE_MC)
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
# =============================================================================
#     df = agama.DistributionFunction(type ='DoublePowerLaw',
#                                     norm=1,
#                                     slopeIn=slopein,
#                                     slopeOut = slopeout,
#                                     J0=J_0,
#                                     coefJrIn = coefJrIn,
#                                     coefJzIn=coefJzIn,
#                                     coefJrOut=coefJrOut,
#                                     coefJzOut=coefJzOut,
#                                     rotFrac = rotFrac,
#                                     Jcutoff=12000.) 
# =============================================================================
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
                                    Jcutoff=50000.)
    
    norm = df.totalMass()
    # Calculate actions
    
    try:
        af = agama.ActionFinder(pot)
    except (RuntimeError):
        print('no actions')
        return -np.inf
    #act_f = af(data,angles=False)
    
    # Calculate likelihood (+ marginalise over unknown coordinates (z,vx,vy))
    # Define the MC samples
# =============================================================================
#     data3d_mc = np.random.uniform(data,data,(NMC,NSTARS,3)) # x,y,vz, in this config => no errors on x,y,vz
#     data_missing_z_mc = np.random.normal(0,1000,(NMC,NSTARS,1)) # z
#     data_missing_vxvy_mc = np.random.normal(0,1000,(NMC,NSTARS,2)) # vx,vy
# =============================================================================
    
# =============================================================================
#     data3d_mc = np.random.normal(data,0.,(NMC,NSTARS,3)) # x,y,vz, in this config => no errors on x,y,vz
#     data_missing_z_mc = np.random.normal(0,200,(NMC,NSTARS,1)) # z
#     data_missing_vxvy_mc = np.random.normal(0,400,(NMC,NSTARS,2)) # vx,vy
# =============================================================================
    
# =============================================================================
#     data_mc = np.dstack((data3d_mc[:,:,0],data3d_mc[:,:,1],
#                          data_missing_z_mc[:,:,0],data_missing_vxvy_mc[:,:,0],
#                          data_missing_vxvy_mc[:,:,1],data3d_mc[:,:,2]))
# =============================================================================
    
    data_mc_flatten      = data_mc.reshape(-1, data_mc.shape[-1])
    acts_mc_flatten    = af(data_mc_flatten,angles=False)
    prob_mc_flatten    = df(acts_mc_flatten)
    index           = np.isnan(prob_mc_flatten)
    prob_mc_flatten[index] = 0.
    prob_mc            = prob_mc_flatten.reshape([NMC,NSTARS])
    #print(prob_mc)
    prob     = np.sum(prob_mc,axis=0)
    #print('__________')
    #print(prob)
    #print('norm:' +str(norm))
    #print('log prob/norm: '+str(np.log(prob/norm)))
    logL = np.sum(np.log(prob/norm)) # assuming all particles have equal mass
    #print('logL: '+str(logL))
    
    
    #logL = np.sum(np.log((df(act_f)/norm)*mass)) # total likelihood (of all the stars in the catalog)
    
    if np.isnan(logL):
        print('nan logL')
        logL = -np.inf
    return (logL)


#%% Apply gradient optimization

initial_guess = np.array([2.5,5.3,7000., 0.8,1.,0.9,0.9,-0.3,1.4*(10**7.),17.,0.8,3.1*(10**10),5.6*(10**10)])
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

#%% Testing

n = 15
slopein_ = np.linspace(0.00001,2.9999,n)
slopeout_ = np.linspace(3.0001,10,n) 
J0_ = np.linspace(3000,15000,n)
Rs_ = np.linspace(5, 30, n)
rho0_  = np.linspace(0.8*10**7.,3*10**7,n)
q_ = np.linspace(0.,1.,n)


post = []
for i in range(n):
    start1 = time.time()
    par_ = np.array([2.5,5.5,8000.,0.75,1.7,0.88,1.1,0.5,1.1*(10**7),17.,0.8,3.1*(10**10),5.6*(10**10)])
    post.append(_posterior(par_))
    print(i)
    end1 = time.time()
    t  = end1-start1
    print('time: '+str(t))
    print('_______')
post = np.array(post)

#%%


par = 'slopein'
param = np.copy(slopein_)
xlabel = r'$\beta_{in}$'

name_plot = 'llhood_'+par+'_nmc_'+str(NMC)+'_nstars_'+str(NSTARS)+'.png'
#name_post_file = 'llhood_'+par+'_nseed_' + str(nseed)+'_nstars_'+str(n_stars)+'.txt'
#name_param_file = par+'_nseed_'+str(nseed)+'_nstars_'+str(n_stars)+'.txt'
label = r'$n_{MC}=$'+str(NMC)+', $n_{*}=$'+str(NSTARS)

#%%

plt.rc('font', family='serif')
plt.scatter(param, post, s=3, c='black', label=label)
plt.plot(param, post, c='black')
plt.xlabel(xlabel, fontsize=12)
plt.ylabel(r'$\mathcal{L}$', fontsize=14)
plt.legend(fontsize=12)

#plt.savefig(name_plot,format='png',bbox_inches='tight')













