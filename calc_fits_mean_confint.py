#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:45:26 2023

@author: Paula G
@description: Calculate mean and 1sigma confidence intervals for MCMC results.
Based on Payel's code.

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import agama

#%%
backend_folder = '~/PhD/Simulations/Auriga/'
backend_file = "Backend_mock6d_nstars100_nseed_1234.h5"


reader = emcee.backends.HDFBackend(backend_file, read_only=True)
samples = reader.get_chain(discard=15000)
log_prob = reader.get_log_prob(flat = True)
labels =np.array([r"$\beta_{in}$", r"$\beta_{out}$", r"$J_{0}$", 'hr','hz','gr','gz','rotFrac',r'$\rho0$', r'$R_{s}$','q',r'M$_{bulge}$',r'M$_{disk}$'])
#%%


samp_flat = reader.get_chain(discard=6000,flat=True)
log_prob_flat_samples  = reader.get_log_prob(discard=6000,flat=True)

# Find best-fit param (13 param)
n_best_fit        = np.where(log_prob_flat_samples==np.max(log_prob_flat_samples))
n_best_fit_single = n_best_fit[0][0]
pars_best = samp_flat[n_best_fit_single,:]


#%%

slopein_best  = pars_best[0]
slopeout_best = pars_best[1]
J0_best       = pars_best[2]
hr_best       = pars_best[3]
hz_best       = pars_best[4]
gr_best       = pars_best[5]
gz_best       = pars_best[6]
rotFrac_best  = pars_best[7]
rho0_best      = pars_best[8]
Rs_best       = pars_best[9]
q_best        = pars_best[10]
massb_best       = pars_best[11]
massd_best        = pars_best[12]


slopein_fits = samp_flat[:,0]
slopeout_fits = samp_flat[:,1]
J0_fits   = samp_flat[:,2]
hr_fits   = samp_flat[:,3]
hz_fits   = samp_flat[:,4]
gr_fits   = samp_flat[:,5]
gz_fits   = samp_flat[:,6]
rotFrac_fits = samp_flat[:,7]
rho0_fits = samp_flat[:,8]
Rs_fits   = samp_flat[:,9]
q_fits   = samp_flat[:,10]
massb_fits       = samp_flat[:,11]
massd_fits        = samp_flat[:,12]

#%% Calculate mean, median and stdev
median     = 50.
onesig_lo  = median-34.1
onesig_hi  = median+34.1


# Mean
slopein_mu = np.mean(slopein_fits)
slopeout_mu = np.mean(slopeout_fits)
J0_mu = np.mean(J0_fits)
hr_mu = np.mean(hr_fits)
hz_mu = np.mean(hz_fits)
gr_mu = np.mean(gr_fits)
gz_mu = np.mean(gz_fits)
rotFrac_mu = np.mean(rotFrac_fits)
rho0_mu = np.mean(rho0_fits)
Rs_mu = np.mean(Rs_fits)
q_mu = np.mean(q_fits)
massb_mu = np.mean(massb_fits)
massd_mu = np.mean(massd_fits)

# Median
slopein_med = np.percentile(slopein_fits,median)
slopeout_med = np.percentile(slopeout_fits,median)
J0_med = np.percentile(J0_fits,median)
hr_med = np.percentile(hr_fits,median)
hz_med = np.percentile(hz_fits,median)
gr_med = np.percentile(gr_fits,median)
gz_med = np.percentile(gz_fits,median)
rotFrac_med = np.percentile(rotFrac_fits,median)
rho0_med = np.percentile(rho0_fits,median)
Rs_med = np.percentile(Rs_fits,median)
q_med = np.percentile(q_fits,median)
massb_med = np.percentile(massb_fits,median)
massd_med = np.percentile(massd_fits,median)

# 1sigma low
slopein_low = np.percentile(slopein_fits,onesig_lo)
slopeout_low = np.percentile(slopeout_fits,onesig_lo)
J0_low = np.percentile(J0_fits,onesig_lo)
hr_low = np.percentile(hr_fits,onesig_lo)
hz_low = np.percentile(hz_fits,onesig_lo)
gr_low = np.percentile(gr_fits,onesig_lo)
gz_low = np.percentile(gz_fits,onesig_lo)
rotFrac_low = np.percentile(rotFrac_fits,onesig_lo)
rho0_low = np.percentile(rho0_fits,onesig_lo)
Rs_low = np.percentile(Rs_fits,onesig_lo)
q_low = np.percentile(q_fits,onesig_lo)
massb_low = np.percentile(massb_fits,onesig_lo)
massd_low = np.percentile(massd_fits,onesig_lo)

# 1sigma high
slopein_hi = np.percentile(slopein_fits,onesig_hi)
slopeout_hi = np.percentile(slopeout_fits,onesig_hi)
J0_hi = np.percentile(J0_fits,onesig_hi)
hr_hi = np.percentile(hr_fits,onesig_hi)
hz_hi = np.percentile(hz_fits,onesig_hi)
gr_hi = np.percentile(gr_fits,onesig_hi)
gz_hi = np.percentile(gz_fits,onesig_hi)
rotFrac_hi = np.percentile(rotFrac_fits,onesig_hi)
rho0_hi = np.percentile(rho0_fits,onesig_hi)
Rs_hi = np.percentile(Rs_fits,onesig_hi)
q_hi = np.percentile(q_fits,onesig_hi)
massb_hi = np.percentile(massb_fits,onesig_hi)
massd_hi = np.percentile(massd_fits,onesig_hi)

#%%

results_mean = np.array([slopein_mu,
                         slopeout_mu,
                         J0_mu,
                         hr_mu,
                         hz_mu,
                         gr_mu,
                         gz_mu,
                         rotFrac_mu,
                         rho0_mu,
                         Rs_mu,
                         q_mu,
                         massb_mu,
                         massd_mu])

results_median = np.array([slopein_med,
                         slopeout_med,
                         J0_med,
                         hr_med,
                         hz_med,
                         gr_med,
                         gz_med,
                         rotFrac_med,
                         rho0_med,
                         Rs_med,
                         q_med,
                         massb_med,
                         massd_med])

results_low = np.array([slopein_low,
                         slopeout_low,
                         J0_low,
                         hr_low,
                         hz_low,
                         gr_low,
                         gz_low,
                         rotFrac_low,
                         rho0_low,
                         Rs_low,
                         q_low,
                         massb_low,
                         massd_low])

results_high = np.array([slopein_hi,
                         slopeout_hi,
                         J0_hi,
                         hr_hi,
                         hz_hi,
                         gr_hi,
                         gz_hi,
                         rotFrac_hi,
                         rho0_hi,
                         Rs_hi,
                         q_hi,
                         massb_hi,
                         massd_hi])

np.savetxt('fits_mock_6d_mean.txt',results_mean)
np.savetxt('fits_mock_6d_median.txt',results_median)
np.savetxt('fits_mock_6d_1siglow.txt',results_low)
np.savetxt('fits_mock_6d_1sighi.txt',results_high)



