#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:16:45 2023

@author: Paula G
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
import binning_methods_new as bm
import CoordTrans as ct

agama.setUnits(mass=1, length=1, velocity=1)

# Plotting settings
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = [4, 3.5]
mpl.rcParams['agg.path.chunksize'] = 10000


# Global parameters
NSTARS = 1000
ROT = 45
SAVE = True


#backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i0.h5"
#backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i90.h5"
backend_file = "Backend_mock_halo_3d_nstars_1000_impsamp_i45.h5"

discard0 = 30000
discard45 = 40000
discard90 = 25000

discard_samp = discard45
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

gm_true = agama.GalaxyModel(pot_true, df_true)


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


# n=2000
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

# %%

# mask = (log_prob_flat_samples>= -30239.4) & (log_prob_flat_samples<=-30232.83) # i=90
mask = (log_prob_flat_samples >= -
        29805) & (log_prob_flat_samples <= -29799.09)  # i=0
# mask = (log_prob_flat_samples>= -30174.4) & (log_prob_flat_samples<=-30168.28) # i=45

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

# %%

n = 700

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

gm_best = agama.GalaxyModel(pot_best, df_best)

# %% Ndens true and best-fit
rgrid_vphi = np.linspace(0.1, 100., 40)
rgrid_vphi_ = np.column_stack((rgrid_vphi, rgrid_vphi*0., rgrid_vphi*0.))


ndens_best = gm_best.moments(rgrid_vphi_, dens=True, vel=False, vel2=False)
ndens_true = gm_true.moments(rgrid_vphi_, dens=True, vel=False, vel2=False)


# %%


ndens_confint = []


for i in range(len(slopein)):
    # Disk potential
    print(i)
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
    pot_conf = agama.Potential(pot_bulge_best, pot_disk_best, pot_dm_conf)
    df_conf = agama.DistributionFunction(type='DoublePowerLaw',
                                         norm=1,
                                         slopeIn=slopein[i],
                                         slopeOut=slopeout[i],
                                         J0=J0[i],
                                         coefJrIn=hr[i],
                                         coefJzIn=hz[i],
                                         coefJrOut=gr[i],
                                         coefJzOut=gz[i],
                                         rotFrac=rotFrac[i],
                                         Jcutoff=20000)

    # Create distribution function instance

    gm_ = agama.GalaxyModel(pot_conf, df_conf)

    ndens_ = gm_.moments(rgrid_vphi_, dens=True, vel=False, vel2=False)
    ndens_confint.append(ndens_)

    print(i)


ndens_confint = np.array(ndens_confint)

# %% Save

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725


ndens_med = []
ndens_upper = []
ndens_lower = []
ndens_2upper = []
ndens_2lower = []

for i in range(len(rgrid_vphi)):
    ndens_med.append(np.percentile(ndens_confint[:, i], median))
    ndens_lower.append(np.percentile(ndens_confint[:, i], onesig_lo))
    ndens_upper.append(np.percentile(ndens_confint[:, i], onesig_hi))
    ndens_2lower.append(np.percentile(ndens_confint[:, i], twosig_lo))
    ndens_2upper.append(np.percentile(ndens_confint[:, i], twosig_hi))


ndens_med = np.array(ndens_med)
ndens_upper = np.array(ndens_upper)
ndens_lower = np.array(ndens_lower)
ndens_2upper = np.array(ndens_2upper)
ndens_2lower = np.array(ndens_2lower)

# %%
if (SAVE):
    np.savetxt('r_ndens_'+str(NSTARS)+'_i'+str(ROT)+'.txt', rgrid_vphi)
    np.savetxt('ndens_true_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_true)
    np.savetxt('ndens_best_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_best)
    np.savetxt('ndens_median_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_med)
    np.savetxt('ndens_upper_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_upper)
    np.savetxt('ndens_lower_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_lower)
    np.savetxt('ndens_2upper_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_2upper)
    np.savetxt('ndens_2lower_'+str(NSTARS)+'_i'+str(ROT)+'.txt', ndens_2lower)

    print('Saved!')


# %%

rgrid_vphi = np.linspace(0.1, 100., 40)
rgrid_vphi_ = np.column_stack((rgrid_vphi, rgrid_vphi*0., rgrid_vphi*0.))


vphi_confint = []


for i in range(len(slopein)):
    # Disk potential
    print(i)
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
    pot_conf = agama.Potential(pot_bulge_best, pot_disk_best, pot_dm_conf)
    df_conf = agama.DistributionFunction(type='DoublePowerLaw',
                                         norm=1,
                                         slopeIn=slopein[i],
                                         slopeOut=slopeout[i],
                                         J0=J0[i],
                                         coefJrIn=hr[i],
                                         coefJzIn=hz[i],
                                         coefJrOut=gr[i],
                                         coefJzOut=gz[i],
                                         rotFrac=rotFrac[i],
                                         Jcutoff=20000)

    # Create distribution function instance

    gm_ = agama.GalaxyModel(pot_conf, df_conf)

    print(i)
    vel_ = gm_.moments(rgrid_vphi_, dens=False, vel=True, vel2=False)
    vphi_confint.append(vel_[:, 1])

vphi_confint = np.array(vphi_confint)


# %% Vphi confidence intervals

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725


vphi_med = []
vphi_upper = []
vphi_lower = []
vphi_2upper = []
vphi_2lower = []

for i in range(len(rgrid_vphi)):
    vphi_med.append(np.percentile(vphi_confint[:, i], median))
    vphi_lower.append(np.percentile(vphi_confint[:, i], onesig_lo))
    vphi_upper.append(np.percentile(vphi_confint[:, i], onesig_hi))
    vphi_2lower.append(np.percentile(vphi_confint[:, i], twosig_lo))
    vphi_2upper.append(np.percentile(vphi_confint[:, i], twosig_hi))


vphi_med = np.array(vphi_med)
vphi_upper = np.array(vphi_upper)
vphi_lower = np.array(vphi_lower)
vphi_2upper = np.array(vphi_2upper)
vphi_2lower = np.array(vphi_2lower)


# %% Vphi

rgrid_vphi = np.linspace(0.1, 100., 40)
rgrid_vphi_ = np.column_stack((rgrid_vphi, rgrid_vphi*0., rgrid_vphi*0.))

vel_best = gm_best.moments(rgrid_vphi_, dens=False, vel=True, vel2=False)
vel_true = gm_true.moments(rgrid_vphi_, dens=False, vel=True, vel2=False)

velphi_best = vel_best[:, 1]
velphi_true = vel_true[:, 1]

# %% Vphi confidence intervals

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725


vphi_med = []
vphi_upper = []
vphi_lower = []
vphi_2upper = []
vphi_2lower = []

for i in range(len(rgrid_vphi)):
    vphi_med.append(np.percentile(vphi_confint[:, i], median))
    vphi_lower.append(np.percentile(vphi_confint[:, i], onesig_lo))
    vphi_upper.append(np.percentile(vphi_confint[:, i], onesig_hi))
    vphi_2lower.append(np.percentile(vphi_confint[:, i], twosig_lo))
    vphi_2upper.append(np.percentile(vphi_confint[:, i], twosig_hi))


vphi_med = np.array(vphi_med)
vphi_upper = np.array(vphi_upper)
vphi_lower = np.array(vphi_lower)
vphi_2upper = np.array(vphi_2upper)
vphi_2lower = np.array(vphi_2lower)

# %%

color = '#b23c6b'

plt.plot(rgrid_vphi, velphi_true, '--', c='black', label='True', linewidth=2)
plt.plot(rgrid_vphi, vphi_med, color=color, linewidth=2, label='Best-fit')
plt.fill_between(rgrid_vphi, vphi_upper, vphi_lower, color=color, alpha=0.2)
plt.fill_between(rgrid_vphi, vphi_2upper, vphi_2lower, color=color, alpha=0.15)

plt.xlabel(r'r (kpc)', fontsize=13)
plt.ylabel(r'$\mathrm{\bar{v}}_{\mathrm{\phi}}$', fontsize=13)
plt.legend(loc='lower right', fontsize=13)
# plt.savefig('vphi_halo_'+str(NSTARS)+'_i'+str(ROT)+'.pdf',format='pdf',bbox_inches='tight')

# %%

if (SAVE):
    np.savetxt('r_vphi_'+str(NSTARS)+'_i'+str(ROT)+'.txt', rgrid_vphi)
    np.savetxt('vphi_true_'+str(NSTARS)+'_i'+str(ROT)+'.txt', velphi_true)
    np.savetxt('vphi_best_'+str(NSTARS)+'_i'+str(ROT)+'.txt', velphi_best)
    np.savetxt('vphi_median_'+str(NSTARS)+'_i'+str(ROT)+'.txt', vphi_med)
    np.savetxt('vphi_upper_'+str(NSTARS)+'_i'+str(ROT)+'.txt', vphi_upper)
    np.savetxt('vphi_lower_'+str(NSTARS)+'_i'+str(ROT)+'.txt', vphi_lower)
    np.savetxt('vphi_2upper_'+str(NSTARS)+'_i'+str(ROT)+'.txt', vphi_2upper)
    np.savetxt('vphi_2lower_'+str(NSTARS)+'_i'+str(ROT)+'.txt', vphi_2lower)

    print('Saved!')

# %% Velocity dispersions (cartesian) true vs best-fit
direction = '010'

r = np.linspace(0.1, 100, 50)

if (direction == '100'):
    rgrid = np.column_stack((r, r*0, r*0))  # 100
if (direction == '010'):
    rgrid = np.column_stack((r*0, r, r*0))  # 010
if (direction == '111'):
    rgrid = np.column_stack((r/np.sqrt(3), r/np.sqrt(3), r/np.sqrt(3)))  # 111


sig_best = gm_best.moments(rgrid, dens=False, vel=False, vel2=True)
sig_true = gm_true.moments(rgrid, dens=False, vel=False, vel2=True)

sigvx_best = np.sqrt(sig_best[:, 0])
sigvy_best = np.sqrt(sig_best[:, 1])
sigvz_best = np.sqrt(sig_best[:, 2])

sigvx_true = np.sqrt(sig_true[:, 0])
sigvy_true = np.sqrt(sig_true[:, 1])
sigvz_true = np.sqrt(sig_true[:, 2])

# %%
if (SAVE):
    np.savetxt('r_disp_cart.txt', r)
    np.savetxt('sig_vx_best_'+direction+'_i'+str(ROT)+'.txt', sigvx_best)
    np.savetxt('sig_vy_best_'+direction+'_i'+str(ROT)+'.txt', sigvy_best)
    np.savetxt('sig_vz_best_'+direction+'_i'+str(ROT)+'.txt', sigvz_best)
    np.savetxt('sig_vx_true_'+direction+'_i'+str(ROT)+'.txt', sigvx_true)
    np.savetxt('sig_vy_true_'+direction+'_i'+str(ROT)+'.txt', sigvy_true)
    np.savetxt('sig_vz_true_'+direction+'_i'+str(ROT)+'.txt', sigvz_true)

# %% Velocity dispersions (cartesian) conf intervals

r = np.linspace(0.1, 100, 50)

if (direction == '100'):
    rgrid = np.column_stack((r, r*0, r*0))  # 100
if (direction == '010'):
    rgrid = np.column_stack((r*0, r, r*0))  # 010
if (direction == '111'):
    rgrid = np.column_stack((r/np.sqrt(3), r/np.sqrt(3), r/np.sqrt(3)))  # 111


sigvx_confint = []
sigvy_confint = []
sigvz_confint = []


for i in range(40):
    # Disk potential
    print(i)
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
    pot_conf = agama.Potential(pot_bulge_best, pot_disk_best, pot_dm_conf)
    df_conf = agama.DistributionFunction(type='DoublePowerLaw',
                                         norm=1,
                                         slopeIn=slopein[i],
                                         slopeOut=slopeout[i],
                                         J0=J0[i],
                                         coefJrIn=hr[i],
                                         coefJzIn=hz[i],
                                         coefJrOut=gr[i],
                                         coefJzOut=gz[i],
                                         rotFrac=rotFrac[i],
                                         Jcutoff=20000)

    # Create distribution function instance

    gm_ = agama.GalaxyModel(pot_conf, df_conf)

    #ndens_ = gm_.moments(rgrid_ndens_,dens = True,vel=False,vel2=False)
    print(i)
    # ndens_confint.append(ndens_)
    sigv_ = gm_.moments(rgrid, dens=False, vel=False, vel2=True)
    sigvx_confint.append(np.sqrt(sigv_[:, 0]))
    sigvy_confint.append(np.sqrt(sigv_[:, 1]))
    sigvz_confint.append(np.sqrt(sigv_[:, 2]))


sigvx_confint = np.array(sigvx_confint)
sigvy_confint = np.array(sigvy_confint)
sigvz_confint = np.array(sigvz_confint)

# %%

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725

# Calculate mean, median and stdev
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


for i in range(len(r)):
    # vx
    sigvx_median.append(np.percentile(sigvx_confint[:, i], median))
    sigvx_upper.append(np.percentile(sigvx_confint[:, i], onesig_hi))
    sigvx_lower.append(np.percentile(sigvx_confint[:, i], onesig_lo))
    sigvx_2upper.append(np.percentile(sigvx_confint[:, i], twosig_hi))
    sigvx_2lower.append(np.percentile(sigvx_confint[:, i], twosig_lo))
    # vy
    sigvy_median.append(np.percentile(sigvy_confint[:, i], median))
    sigvy_upper.append(np.percentile(sigvy_confint[:, i], onesig_hi))
    sigvy_lower.append(np.percentile(sigvy_confint[:, i], onesig_lo))
    sigvy_2upper.append(np.percentile(sigvy_confint[:, i], twosig_hi))
    sigvy_2lower.append(np.percentile(sigvy_confint[:, i], twosig_lo))
    # vz
    sigvz_median.append(np.percentile(sigvz_confint[:, i], median))
    sigvz_upper.append(np.percentile(sigvz_confint[:, i], onesig_hi))
    sigvz_lower.append(np.percentile(sigvz_confint[:, i], onesig_lo))
    sigvz_2upper.append(np.percentile(sigvz_confint[:, i], twosig_hi))
    sigvz_2lower.append(np.percentile(sigvz_confint[:, i], twosig_lo))

sigvx_upper = np.array(sigvx_upper)
sigvy_upper = np.array(sigvy_upper)
sigvz_upper = np.array(sigvz_upper)

sigvx_lower = np.array(sigvx_lower)
sigvy_lower = np.array(sigvy_lower)
sigvz_lower = np.array(sigvz_lower)

sigvx_2upper = np.array(sigvx_2upper)
sigvy_2upper = np.array(sigvy_2upper)
sigvz_2upper = np.array(sigvz_2upper)

sigvx_2lower = np.array(sigvx_2lower)
sigvy_2lower = np.array(sigvy_2lower)
sigvz_2lower = np.array(sigvz_2lower)


# %%

if(SAVE):
    # onesig
    np.savetxt('sigvx_upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvx_upper)
    np.savetxt('sigvy_upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvy_upper)
    np.savetxt('sigvz_upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvz_upper)
    np.savetxt('sigvx_lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvx_lower)
    np.savetxt('sigvy_lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvy_lower)
    np.savetxt('sigvz_lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvz_lower)
    # twosig
    np.savetxt('sigvx_2upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvx_2upper)
    np.savetxt('sigvy_2upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvy_2upper)
    np.savetxt('sigvz_2upper_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvz_2upper)
    np.savetxt('sigvx_2lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvx_2lower)
    np.savetxt('sigvy_2lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvy_2lower)
    np.savetxt('sigvz_2lower_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvz_2lower)
    # median
    np.savetxt('sigvx_median_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvx_median)
    np.savetxt('sigvy_median_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvy_median)
    np.savetxt('sigvz_median_'+direction+'_nstars_' +
               str(NSTARS)+'_i'+str(ROT)+'.txt', sigvz_median)

# %% Vel disp cylindrical (true)

nsamp = int(5*(10**5))
data_true_, _ = gm_true.sample(nsamp)

data_true = bm.truncate_data(data_true_, 200)
pos_disp_cyl, sigvR_true, sigvphi_true, sigvz_true = bm.vel_dispersion_cylindrical(
    data_true, 30)

# %%
if (SAVE):
    np.savetxt('r_disp_cyl.txt', pos_disp_cyl)
    np.savetxt('sig_vR_true_i'+str(ROT)+'.txt', sigvR_true)
    np.savetxt('sig_vphi_true_i'+str(ROT)+'.txt', sigvphi_true)
    np.savetxt('sig_vz_best_i'+str(ROT)+'.txt', sigvz_true)


# %% Velocity dispersion cylindrical coords (sampled)

sigvR_confint = []
sigvphi_confint = []
sigvz_confint = []

nsamp = int(5*(10**5))


for i in range(40):
    # Disk potential
    print(i)

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
    pot_conf = agama.Potential(pot_bulge_true, pot_disk_true, pot_dm_conf)
    df_conf = agama.DistributionFunction(type='DoublePowerLaw',
                                         norm=1,
                                         slopeIn=slopein[i],
                                         slopeOut=slopeout[i],
                                         J0=J0[i],
                                         coefJrIn=hr[i],
                                         coefJzIn=hz[i],
                                         coefJrOut=gr[i],
                                         coefJzOut=gz[i],
                                         rotFrac=rotFrac[i],
                                         Jcutoff=20000)

    # Create distribution function instance

    gm_ = agama.GalaxyModel(pot_conf, df_conf)

    data__, _ = gm_.sample(nsamp)
    data_ = bm.truncate_data(data__, 200)
    pos_conf, sigvR_conf, sigvphi_conf, sigvz_conf = bm.vel_dispersion_cylindrical(
        data_, 30)

    print(i)
    sigvR_confint.append(sigvR_conf)
    sigvphi_confint.append(sigvphi_conf)
    sigvz_confint.append(sigvz_conf)


sigvR_confint = np.array(sigvR_confint)
sigvphi_confint = np.array(sigvphi_confint)
sigvz_confint = np.array(sigvz_confint)

# %% Conf int vel disp cylindrical

median = 50.
onesig_lo = 15.865
onesig_hi = 84.135
twosig_lo = 2.275
twosig_hi = 97.725

# Calculate mean, median and stdev
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


for i in range(len(pos_disp_cyl)):
    # vx
    sigvR_median.append(np.percentile(sigvR_confint[:, i], median))
    sigvR_upper.append(np.percentile(sigvR_confint[:, i], onesig_hi))
    sigvR_lower.append(np.percentile(sigvR_confint[:, i], onesig_lo))
    sigvR_2upper.append(np.percentile(sigvR_confint[:, i], twosig_hi))
    sigvR_2lower.append(np.percentile(sigvR_confint[:, i], twosig_lo))
    # vy
    sigvphi_median.append(np.percentile(sigvphi_confint[:, i], median))
    sigvphi_upper.append(np.percentile(sigvphi_confint[:, i], onesig_hi))
    sigvphi_lower.append(np.percentile(sigvphi_confint[:, i], onesig_lo))
    sigvphi_2upper.append(np.percentile(sigvphi_confint[:, i], twosig_hi))
    sigvphi_2lower.append(np.percentile(sigvphi_confint[:, i], twosig_lo))
    # vz
    sigvz_median.append(np.percentile(sigvz_confint[:, i], median))
    sigvz_upper.append(np.percentile(sigvz_confint[:, i], onesig_hi))
    sigvz_lower.append(np.percentile(sigvz_confint[:, i], onesig_lo))
    sigvz_2upper.append(np.percentile(sigvz_confint[:, i], twosig_hi))
    sigvz_2lower.append(np.percentile(sigvz_confint[:, i], twosig_lo))

sigvR_upper = np.array(sigvR_upper)
sigvphi_upper = np.array(sigvphi_upper)
sigvz_upper = np.array(sigvz_upper)

sigvR_lower = np.array(sigvR_lower)
sigvphi_lower = np.array(sigvphi_lower)
sigvz_lower = np.array(sigvz_lower)

sigvR_2upper = np.array(sigvR_2upper)
sigvphi_2upper = np.array(sigvphi_2upper)
sigvz_2upper = np.array(sigvz_2upper)

sigvR_2lower = np.array(sigvR_2lower)
sigvphi_2lower = np.array(sigvphi_2lower)
sigvz_2lower = np.array(sigvz_2lower)

# %%

if(SAVE):
    # onesig
    np.savetxt('sigvR_upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvR_upper)
    np.savetxt('sigvphi_upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvphi_upper)
    np.savetxt('sigvz_upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvz_upper)
    np.savetxt('sigvR_lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvR_lower)
    np.savetxt('sigvphi_lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvphi_lower)
    np.savetxt('sigvz_lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvz_lower)
    # twosig
    np.savetxt('sigvR_2upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvR_2upper)
    np.savetxt('sigvphi_2upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvphi_2upper)
    np.savetxt('sigvz_2upper_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvz_2upper)
    np.savetxt('sigvR_2lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvR_2lower)
    np.savetxt('sigvphi_2lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvphi_2lower)
    np.savetxt('sigvz_2lower_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvz_2lower)
    # median
    np.savetxt('sigvR_median_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvR_median)
    np.savetxt('sigvphi_median_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvphi_median)
    np.savetxt('sigvz_median_nstars_'+str(NSTARS) +
               '_i'+str(ROT)+'.txt', sigvz_median)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(16, 4)

color0 = '#335c67'
color45 = '#6a994e'
color90 = '#9e2a2b'

color = color0


# ax1.plot(r,sigvx_best,label='Best-fit',c=color)
ax1.plot(pos_disp_cyl, sigvR_median, label='Model', c=color)
ax1.plot(pos_disp_cyl, sigvR_true, '--', label='True', c='black')
ax1.fill_between(pos_disp_cyl, sigvR_upper,
                 sigvR_lower, color=color, alpha=0.3)
ax1.fill_between(pos_disp_cyl, sigvR_2upper,
                 sigvR_2lower, color=color, alpha=0.3)
ax1.legend(fontsize=11)
# ax1.set_xlim((-5,150))
# ax1.set_ylim((50,200.))
ax1.set_xlabel('r (kpc)', fontsize=12)
ax1.set_ylabel(r'$\mathrm{\sigma_{vR} (km/s)}$', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=11)
#ax1.tick_params(axis='both', which='minor', labelsize=11)
# ax1.set_xscale('log')
# ax1.set_yscale('log')


# ax2.plot(r,sigvy_best,label='Best-fit',c=color)
ax2.plot(pos_disp_cyl, sigvphi_median, label='Model', c=color)
ax2.plot(pos_disp_cyl, sigvphi_true, '--', label='True', c='black')
ax2.fill_between(pos_disp_cyl, sigvphi_upper,
                 sigvphi_lower, color=color, alpha=0.3)
ax2.fill_between(pos_disp_cyl, sigvphi_2upper,
                 sigvphi_2lower, color=color, alpha=0.2)
ax2.legend(fontsize=11)
# ax2.set_xlim((-5,150))
# ax2.set_ylim((0,150.))
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_xlabel('r (kpc)', fontsize=12)
ax2.set_ylabel(r'$\mathrm{\sigma_{v\phi} (km/s)}$', fontsize=12)
# ax2.set_xscale('log')
# ax2.set_yscale('log')


# ax3.plot(r,sigvz_best,label='Best-fit',c=color)
ax3.plot(pos_disp_cyl, sigvz_median, label='Model', c=color)
ax3.plot(pos_disp_cyl, sigvz_true, '--', label='True', c='black')
ax3.fill_between(pos_disp_cyl, sigvz_upper,
                 sigvz_lower, color=color, alpha=0.3)
ax3.fill_between(pos_disp_cyl, sigvz_2upper,
                 sigvz_2lower, color=color, alpha=0.2)
# ax3.set_xlim((-5,150))
# ax3.set_ylim((50,175.))
ax3.legend(fontsize=11)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_xlabel('r (kpc)', fontsize=11)
ax3.set_ylabel(r'$\mathrm{\sigma_{vz} (km/s)}$', fontsize=12)
# ax3.set_xscale('log')
# ax3.set_yscale('log')

plt.savefig('veldisp_cyl_halo_'+str(NSTARS)+'_i'+str(ROT) +
            '_FWHM.pdf', format='pdf', bbox_inches='tight')


# %%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(16, 4)
plt.suptitle(r'$\mathrm{\hat{r} = (1,0,0)}$', fontsize=12)

color = '#b23c6b'

# ax1.plot(r,sigvx_best,label='Best-fit',c=color)
ax1.plot(r, sigvx_median, label='Model', c=color)
ax1.plot(r, sigvx_true, '--', label='True', c='black')
ax1.fill_between(r, sigvx_upper, sigvx_lower, color=color, alpha=0.3)
ax1.fill_between(r, sigvx_2upper, sigvx_2lower, color=color, alpha=0.3)
ax1.legend(fontsize=11)
# ax1.set_xlim((-5,150))
# ax1.set_ylim((50,200.))
ax1.set_xlabel('r (kpc)', fontsize=12)
ax1.set_ylabel(r'$\mathrm{\sigma_{vx} (km/s)}$', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=11)
#ax1.tick_params(axis='both', which='minor', labelsize=11)
# ax1.set_xscale('log')
# ax1.set_yscale('log')


# ax2.plot(r,sigvy_best,label='Best-fit',c=color)
ax2.plot(r, sigvy_median, label='Model', c=color)
ax2.plot(r, sigvy_true, '--', label='True', c='black')
ax2.fill_between(r, sigvy_upper, sigvy_lower, color=color, alpha=0.3)
ax2.fill_between(r, sigvy_2upper, sigvy_2lower, color=color, alpha=0.2)
ax2.legend(fontsize=11)
# ax2.set_xlim((-5,150))
# ax2.set_ylim((0,150.))
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.set_xlabel('r (kpc)', fontsize=12)
ax2.set_ylabel(r'$\mathrm{\sigma_{vy} (km/s)}$', fontsize=12)
# ax2.set_xscale('log')
# ax2.set_yscale('log')


# ax3.plot(r,sigvz_best,label='Best-fit',c=color)
ax3.plot(r, sigvz_median, label='Model', c=color)
ax3.plot(r, sigvz_true, '--', label='True', c='black')
ax3.fill_between(r, sigvz_upper, sigvz_lower, color=color, alpha=0.3)
ax3.fill_between(r, sigvz_2upper, sigvz_2lower, color=color, alpha=0.2)
# ax3.set_xlim((-5,150))
# ax3.set_ylim((50,175.))
ax3.legend(fontsize=11)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_xlabel('r (kpc)', fontsize=11)
ax3.set_ylabel(r'$\mathrm{\sigma_{vz} (km/s)}$', fontsize=12)
# ax3.set_xscale('log')
# ax3.set_yscale('log')

# plt.savefig('veldisp_'+direction+'_halo_'+str(NSTARS)+'_i'+str(ROT)+'.pdf',format='pdf',bbox_inches='tight')

# %%

# Velocity dispersions (spherical)
gm_best = agama.GalaxyModel(pot_best, df_best)
data_best_sample, _ = gm_best.sample(100000)

gm_true = agama.GalaxyModel(pot_true, df_true)
data_true_sample, _ = gm_true.sample(100000)

# Dispersions best-fit
dfr_best = bm.calc_disp_beta(data_best_sample, 40)
sig_vr_best = dfr_best['disp_vr']
sig_vtheta_best = dfr_best['disp_vtheta']
sig_vphi_best = dfr_best['disp_vphi']
pos_best = dfr_best['pos']
beta_best = dfr_best['beta']


# Dispersions (true)
dfr_true = bm.calc_disp_beta(data_true_sample, 40)
sig_vr_true = dfr_true['disp_vr']
sig_vtheta_true = dfr_true['disp_vtheta']
sig_vphi_true = dfr_true['disp_vphi']
pos_true = dfr_true['pos']
beta_true = dfr_true['beta']

# %%
if (SAVE):
    np.savetxt('sigvr_true_'+str(NSTARS)+'_i'+str(ROT)+'.txt', sig_vr_true)
    np.savetxt('sigvr_best_'+str(NSTARS)+'_i'+str(ROT)+'.txt', sig_vr_best)
    np.savetxt('sigvtheta_true_'+str(NSTARS)+'_i' +
               str(ROT)+'.txt', sig_vtheta_true)
    np.savetxt('sigvtheta_best_'+str(NSTARS)+'_i' +
               str(ROT)+'.txt', sig_vtheta_best)
    np.savetxt('sigvphi_true_'+str(NSTARS)+'_i'+str(ROT)+'.txt', sig_vphi_true)
    np.savetxt('sigvphi_best_'+str(NSTARS)+'_i'+str(ROT)+'.txt', sig_vphi_best)
    np.savetxt('pos_best_'+str(NSTARS)+'_i'+str(ROT)+'.txt', pos_best)
    np.savetxt('pos_true_'+str(NSTARS)+'_i'+str(ROT)+'.txt', pos_true)
    # np.savetxt('ndens_disk_mu_'+str(NSTARS)+'_i'+str(ROT)+'.txt',ndens_mu)
    # np.savetxt('ndens_disk_med_'+str(NSTARS)+'_i'+str(ROT)+'.txt',ndens_med)
    # np.savetxt('vphi_disk_lo_'+str(NSTARS)+'_i'+str(ROT)+'2.txt',vphi_lo)
    # np.savetxt('vphi_disk_hi_'+str(NSTARS)+'_i'+str(ROT)+'2.txt',vphi_hi)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(20, 6)


rc('font', size=16)
rc('legend', fontsize=13)

# for i in range(n):
#     ax1.plot(pos_[i],disp_vr[i],c='#9cafb7',alpha=0.1)
ax1.plot(pos_best, sig_vr_best, label='Best-fit', c='red')
ax1.plot(pos_true, sig_vr_true, label='True', c='black')
# ax1.plot(pos_mc,sig_vr_mc,label='MC',c='green')
# ax1.plot(pos_hi,sig_vr_hi,label='hi',c='green',alpha=0.4)
# ax1.plot(pos_lo,sig_vr_lo,label='lo',c='green',alpha=0.4)

ax1.legend()
ax1.set_xlim((-10, 100))
# ax1.set_ylim((50,200.))
ax1.set_xlabel('r (kpc)')
ax1.set_ylabel(r'$\sigma_{r} (km/s)$')
# ax1.set_xscale('log')
# ax1.set_yscale('log')

# for i in range(n):
#     ax2.plot(pos_[i],disp_vtheta[i],c='#9cafb7',alpha=0.3)
ax2.plot(pos_best, sig_vtheta_best, label='Best-fit', c='red')
ax2.plot(pos_true, sig_vtheta_true, label='True', c='black')
# ax2.plot(pos_mc,sig_vtheta_mc,label='MC',c='green')
# ax2.plot(pos_hi,sig_vtheta_hi,label='hi',c='green',alpha=0.4)
# ax2.plot(pos_lo,sig_vtheta_lo,label='lo',c='green',alpha=0.4)
ax2.legend()
ax2.set_xlim((-10, 100.))
# ax2.set_ylim((0,150.))
ax2.set_xlabel('r (kpc)')
ax2.set_ylabel(r'$\sigma_{\theta} (km/s)$')
# ax2.set_xscale('log')
# ax2.set_yscale('log')

# for i in range(n):
#     ax3.plot(pos_[i],disp_vphi[i],c='#9cafb7',alpha=0.3)
ax3.plot(pos_best, sig_vphi_best, label='Best-fit', c='red')
ax3.plot(pos_true, sig_vphi_true, label='True', c='black')
# ax3.plot(pos_mc,sig_vphi_mc,label='MC',c='green')
# ax3.plot(pos_hi,sig_vphi_hi,label='hi',c='green',alpha=0.4)
# ax3.plot(pos_lo,sig_vphi_lo,label='lo',c='green',alpha=0.4)

ax3.set_xlim((-10, 100.))
# ax3.set_ylim((50,175.))
ax3.legend()
ax3.set_xlabel('r (kpc)')
ax3.set_ylabel(r'$\sigma_{\phi} (km/s)$')
# ax3.set_xscale('log')
# ax3.set_yscale('log')

fig.suptitle(r'i='+str(ROT)+'$^{\circ}$')

# plt.savefig('sigmas_halo_'+str(NSTARS)+'_i'+str(ROT)+'.pdf',format='pdf',bbox_inches='tight')
