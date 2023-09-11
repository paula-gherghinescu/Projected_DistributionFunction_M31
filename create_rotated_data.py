#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:53:18 2023

@author: paulagherghinescu
"""

import numpy as np
import matplotlib.pyplot as plt
import agama
import CoordTrans as ct
import pickle


# Set units for AGAMA
agama.setUnits(mass=1, length=1, velocity=1)

plt.rc('font', family='serif')


#%% Define rotation operations

def rotyx(data, thetax, thetay):
    thetax_rad = thetax*(np.pi/180)
    thetay_rad = thetay*(np.pi/180)

    cosx = np.cos(thetax_rad)
    sinx = np.sin(thetax_rad)
    cosy = np.cos(thetay_rad)
    siny = np.sin(thetay_rad)

    x = np.copy(data[:, 0])
    y = np.copy(data[:, 1])
    z = np.copy(data[:, 2])
    vx = np.copy(data[:, 3])
    vy = np.copy(data[:, 4])
    vz = np.copy(data[:, 5])

    x_ = x*cosy+y*sinx*siny+z*siny*cosx
    y_ = y*cosx-z*sinx
    z_ = -x*siny+y*sinx*cosy+z*cosx*cosy
    vx_ = vx*cosy+vy*sinx*siny+vz*siny*cosx
    vy_ = vy*cosx-vz*sinx
    vz_ = -vx*siny+vy*sinx*cosy+vz*cosx*cosy

    data_ = np.column_stack((x_, y_, z_, vx_, vy_, vz_))

    return (data_)

def rotyx_inv(data_rot, thetax, thetay):
    thetax_rad = thetax*(np.pi/180)
    thetay_rad = thetay*(np.pi/180)

    cosx = np.cos(thetax_rad)
    sinx = np.sin(thetax_rad)
    cosy = np.cos(thetay_rad)
    siny = np.sin(thetay_rad)

    x = np.copy(data_rot[:, 0])
    y = np.copy(data_rot[:, 1])
    z = np.copy(data_rot[:, 2])
    vx = np.copy(data_rot[:, 3])
    vy = np.copy(data_rot[:, 4])
    vz = np.copy(data_rot[:, 5])

    x_ = x*cosy-z*siny
    y_ = x*sinx*siny + y*cosx+z*sinx*cosy
    z_ = x*siny*cosx-y*sinx+z*cosx*cosy
    vx_ = vx*cosy-vz*siny
    vy_ = vx*sinx*siny + vy*cosx+vz*sinx*cosy
    vz_ = vx*siny*cosx-vy*sinx+vz*cosx*cosy

    data_ = np.column_stack((x_, y_, z_, vx_, vy_, vz_))
    return(data_)

def create_mc_samples(data3d,nstars,nmc):        
    # x,y,vz, in this config => no errors on x,y,vz
    data3d_x_mc = np.random.normal(data3d[:,0], 0., (nmc, nstars))
    data3d_y_mc = np.random.normal(data3d[:,1], 0., (nmc, nstars))
    data3d_vz_mc = np.random.normal(data3d[:,2], 0., (nmc, nstars))
    
    
    data_missing_z_mc = np.random.normal(0, 400, (nmc, nstars))  # z
    data_missing_vx_mc = np.random.normal(0, 400, (nmc, nstars))  # vx,vy
    data_missing_vy_mc = np.random.normal(0, 400, (nmc, nstars))  # vx,vy
    
    
    data_mc = np.dstack((data3d_x_mc, data3d_y_mc,data_missing_z_mc, data_missing_vx_mc,data_missing_vy_mc, data3d_vz_mc))
    
    return (data_mc)

def sample_spherical(nstars):
    # Potential
    rho0 = 1.1*(10**7.)
    Rs = 17.
    q = 0.8
    rdisk = 4.318  # disk scale radius
    mdisk = 6.648*(10**10.)  # disk mass
    rbulge = 0.877  # bulge effective radius
    mbulge = 0.930*(10**10.)
    nb = 1.026  # bulge Sersic index
    r_cut = 1000000.
    
    # Dark Matter potential
    dmHaloNFW_param = dict(type='Spheroid',
                           densityNorm=rho0,
                           gamma=1.,
                           beta=3.,
                           alpha=1.,
                           scaleRadius=Rs,
                           outercutoffradius=r_cut,
                           axisRatioZ=q)
    
    pot_halo = agama.Potential(dmHaloNFW_param)
    
    # Disk potential
    pot_disk = agama.Potential(type='Sersic',
                               mass=mdisk,
                               scaleRadius=rdisk,
                               sersicIndex=1.)
    
    
    # Bulge potential
    pot_bulge = agama.Potential(type='Sersic',
                                mass=mbulge,
                                scaleRadius=rbulge,
                                sersicIndex=nb)
    
    # Total potential
    # total potential of the galaxy
    pot = agama.Potential(pot_bulge, pot_disk, pot_halo)
    
    
    # Distribution function for the stellar halo
    
    df = agama.DistributionFunction(type='DoublePowerLaw',
                                    norm=1,
                                    slopeIn=2.5,
                                    slopeOut=5.5,
                                    J0=8000.,
                                    coefJrIn=0.75,
                                    coefJzIn=1.7,
                                    coefJrOut=0.88,
                                    coefJzOut=1.1,
                                    rotFrac=0.5,
                                    Jcutoff=10**50)
    
    # Galaxy Model object
    gm = agama.GalaxyModel(pot, df)
    data,_ = gm.sample(nstars)
    
    return (data)


#%%

NSTARS = 1000
NMC    = 2000
THETAX = 0
THETAY = 45

data6d = sample_spherical(NSTARS)
data6d_rot = rotyx(data6d,THETAX,THETAY)
data3d_rot = np.column_stack((data6d_rot[:,0],data6d_rot[:,1],data6d_rot[:,5]))
data_rot_mc = create_mc_samples(data3d_rot,NSTARS,NMC)
data_mc_int = rotyx_inv(data_rot_mc,THETAX,THETAY)

#%%

name_file = 'data_mc_nstars_'+str(NSTARS)+'_nmc_'+str(NMC)+'_45rot.pkl'
with open(name_file, 'wb') as f:
    pickle.dump(data_mc_int, f)

print('Saved to...')
print(name_file)










