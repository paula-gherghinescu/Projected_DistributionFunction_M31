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
import utils as ut
import binning_methods as bm
import CoordTrans as ct

agama.setUnits(mass=1, length=1, velocity=1)


# Plotting settings
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = [4, 3.5]
mpl.rcParams['agg.path.chunksize'] = 10000

# Global parameters
NSTARS = 1000
ROT = 0
SAVE = True

backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i0.h5"
#backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i90.h5"
#backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i45.h5"

discard0 = 30000
discard45 = 40000
discard90 = 25000

discard_samp = discard0

#%%

logL_best = np.loadtxt('logL_best_marg_i0.txt')
logL_true = np.loadtxt('logL_true_marg_i0.txt')


# %% Or... get MCMC chains which have been separated

param_best = np.loadtxt('param_fit_mask_1000_i90.txt')
param_best_fit = np.loadtxt('param_fit_mask_1000_i90_best.txt')

# =============================================================================
# param_best = np.loadtxt('param_fit_mask_1000_i45.txt')
# param_best_fit = np.loadtxt('param_fit_mask_1000_i45_best.txt')
# =============================================================================

slopein_best = param_best_fit[0]
slopeout_best = param_best_fit[1]
J0_best = param_best_fit[2]
hr_best = param_best_fit[3]
hz_best = param_best_fit[4]
gr_best = param_best_fit[5]
gz_best = param_best_fit[6]
rotFrac_best = param_best_fit[7]
rho0_best = param_best_fit[8]
Rs_best = param_best_fit[9]
q_best = param_best_fit[10]

slopein_fits = param_best[:, 0]
slopeout_fits = param_best[:, 1]
J0_fits = param_best[:, 2]
hr_fits = param_best[:, 3]
hz_fits = param_best[:, 4]
gr_fits = param_best[:, 5]
gz_fits = param_best[:, 6]
rotFrac_fits = param_best[:, 7]
rho0_fits = param_best[:, 8]
Rs_fits = param_best[:, 9]
q_fits = param_best[:, 10]
# %%

n = 1000

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

print(len(slopein))

# %% True model

# Potential
rho0_true = 1.1*(10**7.)
Rs_true = 17.
q_true = 0.8
rdisk_true = 4.318  # disk scale radius
mdisk_true = 6.648*(10**10.)  # disk mass
rbulge_true = 0.877  # bulge effective radius
mbulge_true = 0.930*(10**10.)
nb_true = 1.026  # bulge Sersic index
r_cut = 1000000.

# Dark Matter potential
dmHaloNFW_param_true = dict(type='Spheroid',
                            densityNorm=rho0_true,
                            gamma=1.,
                            beta=3.,
                            alpha=1.,
                            scaleRadius=Rs_true,
                            outercutoffradius=r_cut,
                            axisRatioZ=q_true)

pot_dm_true = agama.Potential(dmHaloNFW_param_true)

# Disk potential
pot_disk_true = agama.Potential(type='Sersic',
                                mass=mdisk_true,
                                scaleRadius=rdisk_true,
                                sersicIndex=1.)


# Bulge potential
pot_bulge_true = agama.Potential(type='Sersic',
                                 mass=mbulge_true,
                                 scaleRadius=rbulge_true,
                                 sersicIndex=nb_true)

# Total potential
# total potential of the galaxy
pot_true = agama.Potential(pot_bulge_true, pot_disk_true, pot_dm_true)


# Distribution function for the stellar halo

df_true = agama.DistributionFunction(type='DoublePowerLaw',
                                     norm=1,
                                     slopeIn=2.5,
                                     slopeOut=5.5,
                                     J0=8000.,
                                     coefJrIn=0.75,
                                     coefJzIn=1.7,
                                     coefJrOut=0.88,
                                     coefJzOut=1.1,
                                     rotFrac=0.5,
                                     Jcutoff=20000)

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
n = 1000

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

#%%

np.savetxt('param_best_i45.txt',pars_best)


#%%

plt.hist(log_prob_flat_samples,bins=20,color='black',alpha=0.7)
plt.axvline(logL_best,label=r'$\mathrm{\log\mathcal{L}_{best}}$',c='green')
plt.axvline(logL_true,label=r'$\mathrm{\log\mathcal{L}_{true}}$',c='red')
plt.title(r'$\mathrm{i=0^{\circ}}$')

#plt.axvline(x=-3.031347631040872147e+04,c='green')
plt.xlabel(r'$\mathrm{\log \mathcal{L}}$',fontsize=14)
plt.ylabel(r'$\mathrm{N}$',fontsize=14)
plt.legend()

plt.savefig('logL_histogram_i0'+'.pdf', format='pdf', bbox_inches='tight')

#%%

mask = (log_prob_flat_samples>= -30239.4) & (log_prob_flat_samples<=-30232.83) # i=90
#mask = (log_prob_flat_samples>= -29805) & (log_prob_flat_samples<=-29799.09) # i=0
#mask = (log_prob_flat_samples>= -30174.4) & (log_prob_flat_samples<=-30168.28) # i=45


slopein_ = slopein_fits[mask]
slopeout_ = slopeout_fits[mask]
J0_ = J0_fits[mask]
hr_ = hr_fits[mask]
hz_ = hz_fits[mask]
gr_ = gr_fits[mask]
gz_ = gz_fits[mask]
rotFrac_ = rotFrac_fits[mask]
rho0_ = rho0_fits[mask]
Rs_ = Rs_fits[mask]
q_ = q_fits[mask]
print(len(q_))

#%%

n = 300

slopein = slopein_[0::n]
slopeout = slopeout_[0::n]
J0 = J0_[0::n]
hr = hr_[0::n]
hz = hz_[0::n]
gr = gr_[0::n]
gz = gz_[0::n]
rotFrac = rotFrac_[0::n]
rho0 = rho0_[0::n]
Rs = Rs_[0::n]
q = q_[0::n]
print(len(q))


# %% Best-fit model

# Potential (best fit)

# Disk potential
pot_disk_best = agama.Potential(type='Sersic',
                                mass=mdisk_true,
                                scaleRadius=rdisk_true,
                                sersicIndex=1.)


# Bulge potential
pot_bulge_best = agama.Potential(type='Sersic',
                                 mass=mbulge_true,
                                 scaleRadius=rbulge_true,
                                 sersicIndex=nb_true)


# Potential best-fit
pars_DMhalo_best = dict(type='Spheroid',
                        densityNorm=rho0_best,
                        #densityNorm = 1.2*(10**7),
                        #                 gamma=gamma_best,
                        #                 beta=beta_best,
                        #                 alpha=alpha_best,
                        gamma=1.,
                        beta=3.,
                        alpha=1.,
                        scaleRadius=Rs_best,
                        outercutoffradius=1000000.,
                        axisRatioZ=q_best)
pot_dm_best = agama.Potential(**pars_DMhalo_best)
pot_best = agama.Potential(pot_disk_best, pot_bulge_best, pot_dm_best)

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


# %% Total enclosed mass (Agama)
r = np.logspace(np.log10(3), np.log10(100), 100)

M_true = pot_true.enclosedMass(r)
M_best = pot_best.enclosedMass(r)

#%%
if (SAVE):
    np.savetxt('encmass_best_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_best)
    np.savetxt('encmass_true_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_true)


# %%

M_conf = []
# Confidence intervals
for i in range(len(slopein)):
    print(i)
    # Disk potential
    pot_disk_best = agama.Potential(type='Sersic',
                                    mass=mdisk_true,
                                    scaleRadius=rdisk_true,
                                    sersicIndex=1.)

    # Bulge potential
    pot_bulge_best = agama.Potential(type='Sersic',
                                     mass=mbulge_true,
                                     scaleRadius=rbulge_true,
                                     sersicIndex=nb_true)

    # Potential best-fit
    pars_DMhalo_ = dict(type='Spheroid',
                        densityNorm=rho0[i],
                        #densityNorm = 1.2*(10**7),
                        #                 gamma=gamma_best,
                        #                 beta=beta_best,
                        #                 alpha=alpha_best,
                        gamma=1.,
                        beta=3.,
                        alpha=1.,
                        scaleRadius=Rs[i],
                        outercutoffradius=1000000.,
                        axisRatioZ=q[i])
    pot_dm_ = agama.Potential(**pars_DMhalo_)
    pot_conf = agama.Potential(pot_disk_best, pot_bulge_best, pot_dm_)
    M_ = pot_conf.enclosedMass(r)
    M_conf.append(M_)

M_conf = np.array(M_conf)

# %% Total enclosed mass confidence intervals

# Calculate mean, median and stdev

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725


M_upper = []
M_lower = []
M_median = []
M_2upper = []
M_2lower = []

for i in range(len(r)):
    M_upper.append(np.percentile(M_conf[:,i], onesig_hi))
    M_lower.append(np.percentile(M_conf[:,i], onesig_lo))
    M_2upper.append(np.percentile(M_conf[:,i], twosig_hi))
    M_2lower.append(np.percentile(M_conf[:,i], twosig_lo))
    M_median.append(np.percentile(M_conf[:,i], median))


M_upper = np.array(M_upper)
M_lower = np.array(M_lower)
M_2upper = np.array(M_2upper)
M_2lower = np.array(M_2lower)
M_median = np.array(M_median)

#%%

if (SAVE):
    np.savetxt('encmass_upper_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_upper)
    np.savetxt('encmass_lower_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_lower)
    np.savetxt('encmass_2upper_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_2upper)
    np.savetxt('encmass_2lower_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_2lower)
    np.savetxt('encmass_median_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_median)

# %% DM enclosed mass profiles (Agama)

r = np.logspace(np.log10(3), np.log10(100), 100)

M_true_dm = pot_dm_true.enclosedMass(r)
M_best_dm = pot_dm_best.enclosedMass(r)

#%%
if (SAVE):
    np.savetxt('encmass_dm_best_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_best_dm)
    np.savetxt('encmass_dm_true_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_true_dm)


# %% DM confidence intervals

M_conf_dm = []
# Confidence intervals
for i in range(len(slopein)):
    print(i)
    # Disk potential
    pot_disk_best = agama.Potential(type='Sersic',
                                    mass=mdisk_true,
                                    scaleRadius=rdisk_true,
                                    sersicIndex=1.)

    # Bulge potential
    pot_bulge_best = agama.Potential(type='Sersic',
                                     mass=mbulge_true,
                                     scaleRadius=rbulge_true,
                                     sersicIndex=nb_true)

    # Potential best-fit
    pars_DMhalo_ = dict(type='Spheroid',
                        densityNorm=rho0[i],
                        #densityNorm = 1.2*(10**7),
                        #                 gamma=gamma_best,
                        #                 beta=beta_best,
                        #                 alpha=alpha_best,
                        gamma=1.,
                        beta=3.,
                        alpha=1.,
                        scaleRadius=Rs[i],
                        outercutoffradius=1000000.,
                        axisRatioZ=q[i])
    pot_dm_conf = agama.Potential(**pars_DMhalo_)

    M_ = pot_dm_conf.enclosedMass(r)
    M_conf_dm.append(M_)

M_conf_dm = np.array(M_conf_dm)


# %% DM total enclosed mass confidence intervals


median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725


M_dm_upper = []
M_dm_lower = []
M_dm_median = []
M_dm_2lower = []
M_dm_2upper = []

for i in range(len(r)):
    M_dm_upper.append(np.percentile(M_conf_dm[:,i], onesig_hi))
    M_dm_lower.append(np.percentile(M_conf_dm[:,i], onesig_lo))
    M_dm_2upper.append(np.percentile(M_conf_dm[:,i], twosig_hi))
    M_dm_2lower.append(np.percentile(M_conf_dm[:,i], twosig_lo))
    M_dm_median.append(np.percentile(M_conf_dm[:,i], median))


M_dm_upper = np.array(M_dm_upper)
M_dm_lower = np.array(M_dm_lower)
M_dm_2upper = np.array(M_dm_2upper)
M_dm_2lower = np.array(M_dm_2lower)
M_dm_median = np.array(M_dm_median)

#%%

if (SAVE):
    np.savetxt('encmass_dm_upper_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_dm_upper)
    np.savetxt('encmass_dm_lower_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_dm_lower)
    np.savetxt('encmass_dm_2upper_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_dm_2upper)
    np.savetxt('encmass_dm_2lower_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_dm_2lower)
    np.savetxt('encmass_dm_median_halo_'+str(NSTARS)+'_i'+str(ROT)+'.txt', M_dm_median)


# %%
plt.rcParams["figure.figsize"] = [3.5, 2.9]

color0  = '#335c67'
color45 = '#6a994e'
color90 = '#9e2a2b'

color = color45
label = r'Median,i=$45^{\circ}$'


plt.plot(r, M_true, '--', c='black', label='True', linewidth=1.3)
# =============================================================================
# plt.plot(r, M_best, color=color0, linewidth=1.3,
#          label=r'Best-fit,i=$90^{\circ}$')
# =============================================================================
plt.plot(r, M_median, '--', color=color, linewidth=1.3,
         label=label)
plt.fill_between(r, M_upper, M_lower, color=color, alpha=0.3)
plt.fill_between(r, M_2upper, M_2lower, color=color, alpha=0.2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'r (kpc)', fontsize=13)
plt.ylabel(r'$\mathrm{M}_{\mathrm{enc}}$', fontsize=13)
plt.legend(loc='lower right', fontsize=13)
# plt.title(r'N$_{*}=1000$',fontsize=13)
plt.savefig('encmass_halo_'+str(NSTARS)+'_i'+str(ROT) +'_FWHM.pdf', format='pdf', bbox_inches='tight')

# %%
plt.rcParams["figure.figsize"] = [3.5, 2.9]

color = color45

label = label


plt.plot(r, M_true_dm, '--', c='black', label='True', linewidth=2)
# =============================================================================
# plt.plot(r, M_best_dm, color=color0, linewidth=1.3,
#          label=r'Best-fit,i=$90^{\circ}$')
# =============================================================================
plt.plot(r, M_dm_median, color=color, linewidth=1.3,
         label=label)
plt.fill_between(r, M_dm_upper, M_dm_lower, color=color, alpha=0.3)
plt.fill_between(r, M_dm_2upper, M_dm_2lower, color=color, alpha=0.2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'r (kpc)', fontsize=13)
plt.ylabel(r'$\mathrm{M}_{\mathrm{DM,enc}}$', fontsize=13)
plt.legend(loc='lower right', fontsize=13)
# plt.title(r'N$_{*}=1000$',fontsize=13)
plt.savefig('encmass_DM_halo_'+str(NSTARS)+'_i'+str(ROT) +'_FWHM.pdf', format='pdf', bbox_inches='tight')

# %% Plot fractional difference total mass
# =============================================================================
# frac_diff_total = np.abs(M_true-M_best)/M_true
# frac_diff_total_upper = np.abs(M_true-M_upper)/M_true
# frac_diff_total_lower = np.abs(M_true-M_lower)/M_true
# =============================================================================

frac_diff_total = (M_true-M_median)/M_true
frac_diff_total_upper = (M_true-M_upper)/M_true
frac_diff_total_lower = (M_true-M_lower)/M_true
frac_diff_total_2upper = (M_true-M_2upper)/M_true
frac_diff_total_2lower = (M_true-M_2lower)/M_true

zeros = [0]*len(r)

plt.plot(r, frac_diff_total, c=color0, linewidth=1, label=r'$i=90^{\circ}$')
plt.plot(r, zeros, '--',c='grey', linewidth=1, label=r'$i=90^{\circ}$')

plt.fill_between(r, frac_diff_total_upper, frac_diff_total_lower, color=color0, alpha=0.3)
plt.fill_between(r, frac_diff_total_2upper, frac_diff_total_2lower, color=color0, alpha=0.2)

plt.xlabel('r (kpc)', fontsize=13)
plt.ylabel(r'frac diff', fontsize=13)
# plt.yscale('log')
plt.xscale('log')
#plt.ylim((-1, 1))
plt.legend(fontsize=13)
#plt.savefig('fracdiff_encmass_halo_'+str(NSTARS)+'_i'+str(ROT) +'.pdf', format='pdf', bbox_inches='tight')

# %% Plot fractional difference DM
frac_diff_dm = (M_true_dm-M_dm_median)/M_true_dm
frac_diff_dm_upper = (M_true_dm-M_dm_upper)/M_true_dm
frac_diff_dm_lower = (M_true_dm-M_dm_lower)/M_true_dm
frac_diff_dm_2upper = (M_true_dm-M_dm_2upper)/M_true_dm
frac_diff_dm_2lower = (M_true_dm-M_dm_2lower)/M_true_dm
#frac_diff_mc = np.abs(M_Au-M_mc)/M_Au

plt.plot(r, frac_diff_dm, c=color0, linewidth=1, label='Best-fit')
plt.fill_between(r, frac_diff_dm_upper, frac_diff_dm_lower, color=color0, alpha=0.3)
plt.fill_between(r, frac_diff_dm_2upper, frac_diff_dm_2lower, color=color0, alpha=0.3)

# plt.plot(r,frac_diff_mc,c='green',linewidth=1,label='MC')
plt.xlabel('r (kpc)', fontsize=13)
plt.ylabel(r'frac diff', fontsize=13)
# plt.yscale('log')
plt.xscale('log')
#plt.ylim((-1, 1))
plt.legend(fontsize=13)
#plt.savefig('fracdiff_encmass_dm_halo_'+str(NSTARS)+'_i'+str(ROT) +'.pdf', format='pdf', bbox_inches='tight')