#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:30:43 2023

@author: Paula G
@description: Compute confidence intervals for the total and DM enclosed mass.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import pickle
import agama
from scipy.special import gamma as gamma_fct
import pandas as pd
from matplotlib import rc
import matplotlib as mpl
import binning_methods_new as bm

agama.setUnits(mass=1, length=1, velocity=1)


# Plotting settings
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = [4, 3.5]
mpl.rcParams['agg.path.chunksize'] = 10000

# Global parameters
NSTARS = 1000
ROT = 90
SAVE = True
FULL = True

# =============================================================================
# nfull = int(2.5*(10**6.))
# nlowmet = int(1.5*(10**6))
# =============================================================================

nfull = int(5*(10**5))
nlowmet = int(1.5*(10**6))

backend_file = "Backend_Au23_halo_3d_nstars_1000_impsamp_i90_smooth_qprior.h5"
#backend_file = "Backend_Au23_halo_3d_nstars_1000_impsamp_i45_smooth_qprior.h5"
#backend_file = "Backend_Au23_halo_3d_nstars_1000_impsamp_i0_smooth_qprior.h5"





discard0 = 25000
discard45 = 30000
discard90 = 23000

discard_samp = discard90

# =============================================================================
# # %% Or... get MCMC chains which have been separated
# 
# param_best = np.loadtxt('param_fit_mask_1000_i90.txt')
# param_best_fit = np.loadtxt('param_fit_mask_1000_i90_best.txt')
# 
# # =============================================================================
# # param_best = np.loadtxt('param_fit_mask_1000_i45.txt')
# # param_best_fit = np.loadtxt('param_fit_mask_1000_i45_best.txt')
# # =============================================================================
# 
# slopein_best = param_best_fit[0]
# slopeout_best = param_best_fit[1]
# J0_best = param_best_fit[2]
# hr_best = param_best_fit[3]
# hz_best = param_best_fit[4]
# gr_best = param_best_fit[5]
# gz_best = param_best_fit[6]
# rotFrac_best = param_best_fit[7]
# rho0_best = param_best_fit[8]
# Rs_best = param_best_fit[9]
# q_best = param_best_fit[10]
# 
# slopein_fits = param_best[:, 0]
# slopeout_fits = param_best[:, 1]
# J0_fits = param_best[:, 2]
# hr_fits = param_best[:, 3]
# hz_fits = param_best[:, 4]
# gr_fits = param_best[:, 5]
# gz_fits = param_best[:, 6]
# rotFrac_fits = param_best[:, 7]
# rho0_fits = param_best[:, 8]
# Rs_fits = param_best[:, 9]
# q_fits = param_best[:, 10]
# # %%
# 
# n = 1000
# 
# slopein = slopein_fits[0::n]
# slopeout = slopeout_fits[0::n]
# J0 = J0_fits[0::n]
# hr = hr_fits[0::n]
# hz = hz_fits[0::n]
# gr = gr_fits[0::n]
# gz = gz_fits[0::n]
# rotFrac = rotFrac_fits[0::n]
# rho0 = rho0_fits[0::n]
# Rs = Rs_fits[0::n]
# q = q_fits[0::n]
# 
# print(len(slopein))
# 
# =============================================================================
# %% Potential (true)

POTENTIAL_DM   = 'Au23_pot_dm_coef_mul.txt' # coefficients of the multipole potential expansion (Agama)
POTENTIAL_IN   = 'Au23_pot_in_coef_cylsp.txt' # coefficients for the cylindrical spline potential expansion (Agama)
POTENTIAL_ACC  = 'Au23_pot_acc_coef_mul.txt'
POTENTIAL_GAS  = 'Au23_pot_gas_coef_cylsp.txt'

# Define potential
pot_dmAu = agama.Potential(file = POTENTIAL_DM)
pot_inAu = agama.Potential(file = POTENTIAL_IN)
pot_gasAu = agama.Potential(file = POTENTIAL_GAS)
pot_accAu = agama.Potential(file = POTENTIAL_ACC)
pot_totAu = agama.Potential(pot_dmAu,pot_inAu,pot_gasAu,pot_accAu)

#%% Load simulation (accreted)
folder_sim = "/home/paulagherghinescu/PhD/ProjectedDF/MC_sampling/Auriga/Au23_data_acc/"


x_acc = np.loadtxt(folder_sim+'x_acc.txt')
y_acc = np.loadtxt(folder_sim+'y_acc.txt')
z_acc = np.loadtxt(folder_sim+'z_acc.txt')
vx_acc = np.loadtxt(folder_sim+'vx_acc.txt')
vy_acc = np.loadtxt(folder_sim+'vy_acc.txt')
vz_acc = np.loadtxt(folder_sim+'vz_acc.txt')

#%%

xAu = np.copy(x_acc)
yAu = np.copy(y_acc)
zAu = np.copy(z_acc)
vxAu = np.copy(vx_acc)
vyAu = np.copy(vy_acc)
vzAu = np.copy(vz_acc)
rAu = np.sqrt(xAu**2.+yAu**2.+zAu**2.)

dataAu = np.column_stack((xAu,yAu,zAu,vxAu,vyAu,vzAu))

#%% Load smooth halo data

dataAu = np.loadtxt('Au23_lowmet_0.0005-0.001.txt')


# %% Emcee chains + best-fit model

reader = emcee.backends.HDFBackend(backend_file, read_only=True)
samples = reader.get_chain(discard=discard_samp)
log_prob = reader.get_log_prob(flat=True)
labels = np.array([r"$\beta_{in}$", r"$\beta_{out}$",
                  r"$J_{0}$", 'hr', 'hz', 'gr', 'gz',
                   'rotFrac', r'$\rho0$', r'$R_{s}$', 'q'])


# Get best-fit parameters (13 param)
samp_flat = reader.get_chain(discard=discard_samp, flat=True)
log_prob_flat_samples = reader.get_log_prob(discard=discard_samp, flat=True)

# Find best-fit param (13 param)
n_best_fit = np.where(log_prob_flat_samples == np.max(log_prob_flat_samples))
n_best_fit_single = n_best_fit[0][0]
pars_best = samp_flat[n_best_fit_single, :]

slopein_best = pars_best[0]
slopeout_best = pars_best[1]
J0_best = pars_best[2]
hr_best = pars_best[3]
hz_best = pars_best[4]
gr_best = pars_best[5]
gz_best = pars_best[6]
rotFrac_best = pars_best[7]
rho0_best = pars_best[8]
Rs_best = pars_best[9]
q_best = pars_best[10]


slopein_fits = samp_flat[:, 0]
slopeout_fits = samp_flat[:, 1]
J0_fits = samp_flat[:, 2]
hr_fits = samp_flat[:, 3]
hz_fits = samp_flat[:, 4]
gr_fits = samp_flat[:, 5]
gz_fits = samp_flat[:, 6]
rotFrac_fits = samp_flat[:, 7]
rho0_fits = samp_flat[:, 8]
Rs_fits = samp_flat[:, 9]
q_fits = samp_flat[:, 10]


# n=22200
n = 500

slopein = slopein_fits[0::n]
slopeout = slopeout_fits[0::n]
J0 = J0_fits[0::n]
hr = hr_fits[0::n]
hz = hz_fits[0::n]
gr = gr_fits[0::n]
gz = gz_fits[0::n]
rotFrac = rotFrac_fits[0::n]
rho0 = rho0_fits[0::n]
Rs = Rs_fits[0::n]
q = q_fits[0::n]
print(len(q))



#%% Best-fit model

# Disk and bulge potentials (fixed)
Rd = 5.799  # disk scale radius
Md = 6.804*(10**10.)  # disk mass
Rb = 0.871  # bulge effective radius
Mb = 1.442 *(10**10.)
nb = 1.021   # bulge Sersic index

POT_BULGE = agama.Potential(type='Sersic',
                            mass=Mb,
                            scaleRadius=Rb,
                            sersicIndex=nb)

POT_DISK = agama.Potential(type='Sersic',
                           mass=Md,
                           scaleRadius=Rd,
                           sersicIndex=1.)
# Potential best-fit
pars_DMhalo_best = dict(type='Spheroid',
                densityNorm=rho0_best,
                #densityNorm = 1.2*(10**7),
#                 gamma=gamma_best,
#                 beta=beta_best,
#                 alpha=alpha_best,
                        gamma = 1.,
                        beta = 3.,
                        alpha=1.,
                scaleRadius = Rs_best,
                outercutoffradius=1000000.,
                axisRatioZ = q_best)
pot_dm_best = agama.Potential(**pars_DMhalo_best)
pot_best = agama.Potential(POT_DISK,POT_BULGE,pot_dm_best)

# Dsitribution function (best fit)
df_best = agama.DistributionFunction(type='DoublePowerLaw',
                                     norm=1,
                                     slopeIn=slopein_best,
                                     slopeOut=slopeout_best,
                                     J0=J0_best,
                                     coefJrIn=hr_best,
                                     coefJzIn=hz_best,
                                     coefJrOut=gr_best,
                                     coefJzOut=gz_best,
                                     rotFrac=rotFrac_best,
                                     Jcutoff=20000)

#%% True vel dispersion cartesian

pos_true,sigvx_true,sigvy_true,sigvz_true = bm.vel_dispersion_cartesian_error(dataAu,30,False)

#%%

np.savetxt('pos_true_sig_lowmet.txt', pos_true)
np.savetxt('sigvx_true_sig_lowmet.txt', sigvx_true)
np.savetxt('sigvy_true_sig_lowmet.txt', sigvy_true)
np.savetxt('sigvz_true_sig_lowmet.txt', sigvz_true)


#%% True vel dispersion cyldindrical

pos_true_cyl,sigvR_true,sigvphi_true,sigvz_true = bm.vel_dispersion_cylindrical(dataAu,30)

#%%

# =============================================================================
# np.savetxt('pos_true_sig_cyl.txt', pos_true_cyl)
# np.savetxt('sigvR_true.txt', sigvR_true)
# np.savetxt('sigvphi_true.txt', sigvphi_true)
# np.savetxt('sigvz_true.txt', sigvz_true)
# =============================================================================
#%%
np.savetxt('pos_true_sig_cyl_lowmet.txt', pos_true_cyl)
np.savetxt('sigvR_true_lowmet.txt', sigvR_true)
np.savetxt('sigvphi_true_lowmet.txt', sigvphi_true)
np.savetxt('sigvz_true_lowmet.txt', sigvz_true)


#%% 
if(FULL):
    nsamp = nfull
else:
    nsamp = nlowmet



gm_best = agama.GalaxyModel(pot_best,df_best)
data_best_,_ = gm_best.sample(nsamp)

data_best = bm.truncate_data(data_best_,max(rAu))

pos_best,sigvx_best,sigvy_best,sigvz_best = bm.vel_dispersion_cartesian_error(data_best,30,False)

#%%

np.savetxt('pos_best_sig_i'+str(ROT)+'.txt', pos_best)
np.savetxt('sigvx_best_sig_i'+str(ROT)+'.txt', sigvx_best)
np.savetxt('sigvy_best_sig_i'+str(ROT)+'.txt', sigvy_best)
np.savetxt('sigvz_best_sig_i'+str(ROT)+'.txt', sigvz_best)


#%% Confidence intervals

sigvx_confint = []
sigvy_confint = []
sigvz_confint = []


Rd = 5.799  # disk scale radius
Md = 6.804*(10**10.)  # disk mass
Rb = 0.871  # bulge effective radius
Mb = 1.442 *(10**10.)
nb = 1.021   # bulge Sersic index

POT_BULGE = agama.Potential(type='Sersic',
                            mass=Mb,
                            scaleRadius=Rb,
                            sersicIndex=nb)

POT_DISK = agama.Potential(type='Sersic',
                           mass=Md,
                           scaleRadius=Rd,
                           sersicIndex=1.)


for i in range(50):
    # Disk potential
    print(i)
    
    # Potential best-fit
    pars_DMhalo_ = dict(type='Spheroid',
                    densityNorm=rho0[i],
                    #densityNorm = 1.2*(10**7),
    #                 gamma=gamma_best,
    #                 beta=beta_best,
    #                 alpha=alpha_best,
                            gamma = 1.,
                            beta = 3.,
                            alpha=1.,
                    scaleRadius = Rs[i],
                    outercutoffradius=1000000.,
                    axisRatioZ = q[i])
    
    pot_dm_conf = agama.Potential(**pars_DMhalo_)
    pot_conf    = agama.Potential(POT_BULGE,POT_DISK,pot_dm_conf)
    df_conf = agama.DistributionFunction(type ='DoublePowerLaw',
                                    norm=1,
                                    slopeIn=slopein[i],
                                    slopeOut = slopeout[i],
                                    J0=J0[i],
                                    coefJrIn = hr[i],
                                    coefJzIn=hz[i],
                                    coefJrOut=gr[i],
                                    coefJzOut=gz[i],
                                    rotFrac = rotFrac[i],
                                    Jcutoff=20000) 

    # Create distribution function instance
    
    gm_ = agama.GalaxyModel(pot_conf,df_conf)
    
    data__,_ = gm_.sample(nsamp)
    data_ = bm.truncate_data(data__,max(rAu))
    pos_conf,sigvx_conf,sigvy_conf,sigvz_conf = bm.vel_dispersion_cartesian_error(data_,30,False)

    print(i)
    sigvx_confint.append(sigvx_conf)
    sigvy_confint.append(sigvy_conf)
    sigvz_confint.append(sigvz_conf)
    
    
sigvx_confint = np.array(sigvx_confint)
sigvy_confint = np.array(sigvy_confint)
sigvz_confint = np.array(sigvz_confint)

#%%

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725

#Calculate mean, median and stdev
sigvx_median = []
sigvy_median = []
sigvz_median = []


sigvx_upper = []
sigvy_upper = []
sigvz_upper = []

sigvx_lower = []
sigvy_lower = []
sigvz_lower = []


sigvx_2upper = []
sigvy_2upper = []
sigvz_2upper = []

sigvx_2lower = []
sigvy_2lower = []
sigvz_2lower = []


for i in range(len(pos_best)):
    #vx
    sigvx_median.append(np.percentile(sigvx_confint[:, i], median))
    sigvx_upper.append(np.percentile(sigvx_confint[:, i], onesig_hi))
    sigvx_lower.append(np.percentile(sigvx_confint[:, i], onesig_lo))
    sigvx_2upper.append(np.percentile(sigvx_confint[:, i], twosig_hi))
    sigvx_2lower.append(np.percentile(sigvx_confint[:, i], twosig_lo))
    #vy
    sigvy_median.append(np.percentile(sigvy_confint[:, i], median))
    sigvy_upper.append(np.percentile(sigvy_confint[:, i], onesig_hi))
    sigvy_lower.append(np.percentile(sigvy_confint[:, i], onesig_lo))
    sigvy_2upper.append(np.percentile(sigvy_confint[:, i], twosig_hi))
    sigvy_2lower.append(np.percentile(sigvy_confint[:, i], twosig_lo))
    #vz
    sigvz_median.append(np.percentile(sigvz_confint[:, i], median))
    sigvz_upper.append(np.percentile(sigvz_confint[:, i], onesig_hi))
    sigvz_lower.append(np.percentile(sigvz_confint[:, i], onesig_lo))
    sigvz_2upper.append(np.percentile(sigvz_confint[:, i], twosig_hi))
    sigvz_2lower.append(np.percentile(sigvz_confint[:, i], twosig_lo))
    
    
    
sigvx_median = np.array(sigvx_median)
sigvy_median =  np.array(sigvy_median)
sigvz_median =  np.array(sigvz_median)

sigvx_upper = np.array(sigvx_upper)
sigvy_upper =  np.array(sigvy_upper)
sigvz_upper =  np.array(sigvz_upper)

sigvx_lower = np.array(sigvx_lower)
sigvy_lower = np.array(sigvy_lower)
sigvz_lower = np.array(sigvz_lower)

sigvx_2upper = np.array(sigvx_2upper)
sigvy_2upper =  np.array(sigvy_2upper)
sigvz_2upper =  np.array(sigvz_2upper)

sigvx_2lower = np.array(sigvx_2lower)
sigvy_2lower = np.array(sigvy_2lower)
sigvz_2lower = np.array(sigvz_2lower)

#%%

if(SAVE):
    # onesig
    np.savetxt('sigvx_Au23_upper_i'+str(ROT)+'.txt',sigvx_upper)
    np.savetxt('sigvy_Au23_upper_i'+str(ROT)+'.txt',sigvy_upper)
    np.savetxt('sigvz_Au23_upper_i'+str(ROT)+'.txt',sigvz_upper)
    np.savetxt('sigvx_Au23_lower_i'+str(ROT)+'.txt',sigvx_lower)
    np.savetxt('sigvy_Au23_lower_i'+str(ROT)+'.txt',sigvy_lower)
    np.savetxt('sigvz_Au23_lower_i'+str(ROT)+'.txt',sigvz_lower)
    # twosig
    np.savetxt('sigvx_Au23_2upper_i'+str(ROT)+'.txt',sigvx_2upper)
    np.savetxt('sigvy_Au23_2upper_i'+str(ROT)+'.txt',sigvy_2upper)
    np.savetxt('sigvz_Au23_2upper_i'+str(ROT)+'.txt',sigvz_2upper)
    np.savetxt('sigvx_Au23_2lower_i'+str(ROT)+'.txt',sigvx_2lower)
    np.savetxt('sigvy_Au23_2lower_i'+str(ROT)+'.txt',sigvy_2lower)
    np.savetxt('sigvz_Au23_2lower_i'+str(ROT)+'.txt',sigvz_2lower)
    
    # median
    np.savetxt('sigvx_median_i'+str(ROT)+'.txt',sigvx_median)
    np.savetxt('sigvy_median_i'+str(ROT)+'.txt',sigvy_median)
    np.savetxt('sigvz_median_i'+str(ROT)+'.txt',sigvz_median)

#%% Vel dispersion for cyldindrical coordinates




#%% Confidence intervals cylindrical coords

nsamp = 5*(10**5)

sigvR_confint = []
sigvphi_confint = []
sigvz_confint = []


Rd = 5.799  # disk scale radius
Md = 6.804*(10**10.)  # disk mass
Rb = 0.871  # bulge effective radius
Mb = 1.442 *(10**10.)
nb = 1.021   # bulge Sersic index

POT_BULGE = agama.Potential(type='Sersic',
                            mass=Mb,
                            scaleRadius=Rb,
                            sersicIndex=nb)

POT_DISK = agama.Potential(type='Sersic',
                           mass=Md,
                           scaleRadius=Rd,
                           sersicIndex=1.)


for i in range(50):
    # Disk potential
    print(i)
    
    # Potential best-fit
    pars_DMhalo_ = dict(type='Spheroid',
                    densityNorm=rho0[i],
                    #densityNorm = 1.2*(10**7),
    #                 gamma=gamma_best,
    #                 beta=beta_best,
    #                 alpha=alpha_best,
                            gamma = 1.,
                            beta = 3.,
                            alpha=1.,
                    scaleRadius = Rs[i],
                    outercutoffradius=1000000.,
                    axisRatioZ = q[i])
    
    pot_dm_conf = agama.Potential(**pars_DMhalo_)
    pot_conf    = agama.Potential(POT_BULGE,POT_DISK,pot_dm_conf)
    df_conf = agama.DistributionFunction(type ='DoublePowerLaw',
                                    norm=1,
                                    slopeIn=slopein[i],
                                    slopeOut = slopeout[i],
                                    J0=J0[i],
                                    coefJrIn = hr[i],
                                    coefJzIn=hz[i],
                                    coefJrOut=gr[i],
                                    coefJzOut=gz[i],
                                    rotFrac = rotFrac[i],
                                    Jcutoff=20000) 

    # Create distribution function instance
    
    gm_ = agama.GalaxyModel(pot_conf,df_conf)
    
    data__,_ = gm_.sample(nsamp)
    data_ = bm.truncate_data(data__,max(rAu))
    pos_conf,sigvR_conf,sigvphi_conf,sigvz_conf = bm.vel_dispersion_cylindrical(data_,30)

    print(i)
    sigvR_confint.append(sigvR_conf)
    sigvphi_confint.append(sigvphi_conf)
    sigvz_confint.append(sigvz_conf)
    
    
sigvR_confint = np.array(sigvR_confint)
sigvphi_confint = np.array(sigvphi_confint)
sigvz_confint = np.array(sigvz_confint)

#%% Conf int vel disp cylindrical

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725

#Calculate mean, median and stdev
sigvR_median = []
sigvphi_median = []
sigvz_median = []


sigvR_upper = []
sigvphi_upper = []
sigvz_upper = []

sigvR_lower = []
sigvphi_lower = []
sigvz_lower = []


sigvR_2upper = []
sigvphi_2upper = []
sigvz_2upper = []

sigvR_2lower = []
sigvphi_2lower = []
sigvz_2lower = []


for i in range(len(pos_conf)):
    #vx
    sigvR_median.append(np.percentile(sigvR_confint[:, i], median))
    sigvR_upper.append(np.percentile(sigvR_confint[:, i], onesig_hi))
    sigvR_lower.append(np.percentile(sigvR_confint[:, i], onesig_lo))
    sigvR_2upper.append(np.percentile(sigvR_confint[:, i], twosig_hi))
    sigvR_2lower.append(np.percentile(sigvR_confint[:, i], twosig_lo))
    #vy
    sigvphi_median.append(np.percentile(sigvphi_confint[:, i], median))
    sigvphi_upper.append(np.percentile(sigvphi_confint[:, i], onesig_hi))
    sigvphi_lower.append(np.percentile(sigvphi_confint[:, i], onesig_lo))
    sigvphi_2upper.append(np.percentile(sigvphi_confint[:, i], twosig_hi))
    sigvphi_2lower.append(np.percentile(sigvphi_confint[:, i], twosig_lo))
    #vz
    sigvz_median.append(np.percentile(sigvz_confint[:, i], median))
    sigvz_upper.append(np.percentile(sigvz_confint[:, i], onesig_hi))
    sigvz_lower.append(np.percentile(sigvz_confint[:, i], onesig_lo))
    sigvz_2upper.append(np.percentile(sigvz_confint[:, i], twosig_hi))
    sigvz_2lower.append(np.percentile(sigvz_confint[:, i], twosig_lo))

sigvR_upper = np.array(sigvR_upper)
sigvphi_upper =  np.array(sigvphi_upper)
sigvz_upper =  np.array(sigvz_upper)

sigvR_lower = np.array(sigvR_lower)
sigvphi_lower = np.array(sigvphi_lower)
sigvz_lower = np.array(sigvz_lower)

sigvR_2upper = np.array(sigvR_2upper)
sigvphi_2upper =  np.array(sigvphi_2upper)
sigvz_2upper =  np.array(sigvz_2upper)

sigvR_2lower = np.array(sigvR_2lower)
sigvphi_2lower = np.array(sigvphi_2lower)
sigvz_2lower = np.array(sigvz_2lower)

#%%
if(SAVE):
    # onesig
    np.savetxt('sigvR_Au23_upper_i'+str(ROT)+'.txt',sigvR_upper)
    np.savetxt('sigvphi_Au23_upper_i'+str(ROT)+'.txt',sigvphi_upper)
    np.savetxt('sigvz_Au23_upper_i'+str(ROT)+'.txt',sigvz_upper)
    np.savetxt('sigvR_Au23_lower_i'+str(ROT)+'.txt',sigvR_lower)
    np.savetxt('sigvphi_Au23_lower_i'+str(ROT)+'.txt',sigvphi_lower)
    np.savetxt('sigvz_Au23_lower_i'+str(ROT)+'.txt',sigvz_lower)
    # twosig
    np.savetxt('sigvR_Au23_2upper_i'+str(ROT)+'.txt',sigvR_2upper)
    np.savetxt('sigvphi_Au23_2upper_i'+str(ROT)+'.txt',sigvphi_2upper)
    np.savetxt('sigvz_Au23_2upper_i'+str(ROT)+'.txt',sigvz_2upper)
    np.savetxt('sigvR_Au23_2lower_i'+str(ROT)+'.txt',sigvR_2lower)
    np.savetxt('sigvphi_Au23_2lower_i'+str(ROT)+'.txt',sigvphi_2lower)
    np.savetxt('sigvz_Au23_2lower_i'+str(ROT)+'.txt',sigvz_2lower)
    
    # median
    np.savetxt('sigvR_median_i'+str(ROT)+'.txt',sigvR_median)
    np.savetxt('sigvphi_median_i'+str(ROT)+'.txt',sigvphi_median)
    np.savetxt('sigvz_median_i'+str(ROT)+'.txt',sigvz_median)




#%% 

fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
fig.set_size_inches(16,4)
#plt.suptitle(r'$\mathrm{\hat{r} = (1,0,0)}$',fontsize=12)

color = '#b23c6b'

#ax1.plot(r,sigvx_best,label='Best-fit',c=color)
ax1.plot(pos_best,sigvx_median,label='Model',c=color)
ax1.plot(pos_best,sigvx_true,'--',label='True',c='black')
ax1.fill_between(pos_best,sigvx_upper,sigvx_lower,color=color,alpha=0.3)
ax1.fill_between(pos_best,sigvx_2upper,sigvx_2lower,color=color,alpha=0.3)
ax1.legend(fontsize=11)
#ax1.set_xlim((-5,150))
#ax1.set_ylim((50,200.))
ax1.set_xlabel('r (kpc)',fontsize=12)
ax1.set_ylabel(r'$\mathrm{\sigma_{vx} (km/s)}$',fontsize=12)#
ax1.tick_params(axis='both', which='major', labelsize=11)
#ax1.tick_params(axis='both', which='minor', labelsize=11)
#ax1.set_xscale('log')
#ax1.set_yscale('log')


#ax2.plot(r,sigvy_best,label='Best-fit',c=color)
ax2.plot(pos_best,sigvy_median,label='Model',c=color)
ax2.plot(pos_best,sigvy_true,'--',label='True',c='black')
ax2.fill_between(pos_best,sigvy_upper,sigvy_lower,color=color,alpha=0.3)
ax2.fill_between(pos_best,sigvy_2upper,sigvy_2lower,color=color,alpha=0.2)
ax2.legend(fontsize=11)
#ax2.set_xlim((-5,150))
#ax2.set_ylim((0,150.))
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_xlabel('r (kpc)',fontsize=12)
ax2.set_ylabel(r'$\mathrm{\sigma_{vy} (km/s)}$',fontsize=12)
#ax2.set_xscale('log')
#ax2.set_yscale('log')


#ax3.plot(r,sigvz_best,label='Best-fit',c=color)
ax3.plot(pos_best,sigvz_median,label='Model',c=color)
ax3.plot(pos_best,sigvz_true,'--',label='True',c='black')
ax3.fill_between(pos_best,sigvz_upper,sigvz_lower,color=color,alpha=0.3)
ax3.fill_between(pos_best,sigvz_2upper,sigvz_2lower,color=color,alpha=0.2)
#ax3.set_xlim((-5,150))
#ax3.set_ylim((50,175.))
ax3.legend(fontsize=11)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_xlabel('r (kpc)',fontsize=11)
ax3.set_ylabel(r'$\mathrm{\sigma_{vz} (km/s)}$',fontsize=12)
#ax3.set_xscale('log')
#ax3.set_yscale('log')
    
plt.savefig('veldisp_Au23_i'+str(ROT)+'.pdf',format='pdf',bbox_inches='tight')

#%% 

fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
fig.set_size_inches(16,4)
#plt.suptitle(r'$\mathrm{\hat{r} = (1,0,0)}$',fontsize=12)

color = '#b23c6b'

#ax1.plot(r,sigvx_best,label='Best-fit',c=color)
ax1.plot(pos_conf,sigvR_median,label='Model',c=color)
ax1.plot(pos_true_cyl,sigvR_true,'--',label='True',c='black')
ax1.fill_between(pos_conf,sigvR_upper,sigvR_lower,color=color,alpha=0.3)
ax1.fill_between(pos_conf,sigvR_2upper,sigvR_2lower,color=color,alpha=0.3)
ax1.legend(fontsize=11)
#ax1.set_xlim((-5,150))
#ax1.set_ylim((50,200.))
ax1.set_xlabel('r (kpc)',fontsize=12)
ax1.set_ylabel(r'$\mathrm{\sigma_{vx} (km/s)}$',fontsize=12)#
ax1.tick_params(axis='both', which='major', labelsize=11)
#ax1.tick_params(axis='both', which='minor', labelsize=11)
#ax1.set_xscale('log')
#ax1.set_yscale('log')


#ax2.plot(r,sigvy_best,label='Best-fit',c=color)
ax2.plot(pos_conf,sigvphi_median,label='Model',c=color)
ax2.plot(pos_true_cyl,sigvphi_true,'--',label='True',c='black')
ax2.fill_between(pos_conf,sigvphi_upper,sigvphi_lower,color=color,alpha=0.3)
ax2.fill_between(pos_conf,sigvphi_2upper,sigvphi_2lower,color=color,alpha=0.2)
ax2.legend(fontsize=11)
#ax2.set_xlim((-5,150))
#ax2.set_ylim((0,150.))
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_xlabel('r (kpc)',fontsize=12)
ax2.set_ylabel(r'$\mathrm{\sigma_{vy} (km/s)}$',fontsize=12)
#ax2.set_xscale('log')
#ax2.set_yscale('log')


ax3.plot(pos_conf,sigvz_median,label='Model',c=color)
ax3.plot(pos_true_cyl,sigvz_true,'--',label='True',c='black')
ax3.fill_between(pos_conf,sigvz_upper,sigvz_lower,color=color,alpha=0.3)
ax3.fill_between(pos_conf,sigvz_2upper,sigvz_2lower,color=color,alpha=0.2)
#ax3.set_xlim((-5,150))
#ax3.set_ylim((50,175.))
ax3.legend(fontsize=11)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_xlabel('r (kpc)',fontsize=11)
ax3.set_ylabel(r'$\mathrm{\sigma_{vz} (km/s)}$',fontsize=12)
#ax3.set_xscale('log')
#ax3.set_yscale('log')
    
plt.savefig('veldisp_cyl_Au23_i'+str(ROT)+'.pdf',format='pdf',bbox_inches='tight')
