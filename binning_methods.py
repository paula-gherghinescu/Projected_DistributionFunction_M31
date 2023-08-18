#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:57:40 2022

@author: paulagherghinescu
@description: useful functions for binning data and calculating density profiles, velocity dispersions etc.

"""
# Imports 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import CoordTrans as ct
import scipy


# =============================================================================
# def ndensity(data,nbins):
#     """
#     Computes the number density profile through binning.
#     Inputs:
#         data = np.ndarray. Positions (x,y,z). The data will be binned in radii bins with r = sqrt(x^2+y^2+z^2).
#         rmin = min radius of bins. the data points with r<rmin will be ignored
#         rmax
#     """
#     x = data[:,0]
#     y = data[:,1]
#     z = data[:,2]
#     r = np.sqrt(x**2.+y**2.+z**2.)
#     r = r[r>1.]
#     bins = np.logspace(np.log10(min(r)),np.log10(max(r)),nbins)
#     
#     N = plt.hist(r,bins=bins,fill=False, linewidth=0)[0]
#     aux = plt.hist(r,bins=bins,fill=False, linewidth=0)[1]
#     pos = np.zeros((len(aux)-1,))
#     for i in range(len(aux)-1):
#         pos[i] = 0.5*(aux[i]+aux[i+1])
#     delta_pos = np.diff(aux)
#     dens = N/(4*np.pi*(pos**2.)*delta_pos)
#     return(dens,pos)
# =============================================================================

def ndensity(data,nbins):
    """
    Computes the number density profile through binning.
    Inputs:
        data = np.ndarray. Positions (x,y,z). The data will be binned in radii bins with r = sqrt(x^2+y^2+z^2).
        rmin = min radius of bins. the data points with r<rmin will be ignored
        rmax
    """
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    r = np.sqrt(x**2.+y**2.+z**2.)
    #r = r[r>0.1]
    #bins = np.logspace(np.log10(min(r)),np.log10(max(r)),nbins)
    bins = np.logspace(np.log10(0.001),np.log10(250),nbins)
    
    N = plt.hist(r,bins=bins,fill=False, linewidth=0)[0]
    aux = plt.hist(r,bins=bins,fill=False, linewidth=0)[1]
    pos = [np.sqrt(aux[i]*aux[i+1]) for i in range(len(bins)-1)]
    pos = np.array(pos)
    dens = [(3*N[i]/(4*np.pi))*((aux[i+1]**3-aux[i]**3)**(-1)) for i in range(len(aux)-1)]
    dens = np.array(dens)
    return(dens,pos)


def calc_disp_beta(data,nbins):
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    r = r[r>0.001]
    
    rmax = max(r)
    rmin = min(r)
   # print('rmin: '+str(rmin))

    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)


    # Create r bins
    log_rmin = np.log10(rmin) #logspace
    log_rmax = np.log10(rmax) #logspace
    n =nbins
    bins = np.logspace(log_rmin,log_rmax,n)
    h = np.histogram(r,bins=bins)
    r_binIndex = np.digitize(data_sph[:,0], bins)
    
    dict_dataSph = {'r':data_sph[:,0],
                'theta':data_sph[:,1],
                'phi': data_sph[:,2],
                'vr': data_sph[:,3],
                'vtheta': data_sph[:,4],
                'vphi': data_sph[:,5],
                'r_binIndex': r_binIndex}
    df_dataSph   = pd.DataFrame(dict_dataSph) # Data frame with positions r and their corresponding bin
    df_pos       = df_dataSph.groupby(['r_binIndex']).agg(pos = ('r',np.mean))    
    pos          = np.array(df_pos['pos'])
    df_disp = df_dataSph.groupby(['r_binIndex']).agg(
        disp_vr = ("vr", np.std),
        disp_vtheta = ("vtheta", np.std),
        disp_vphi = ("vphi", np.std))
    sig_vr = np.array(df_disp['disp_vr'])
    sig_vtheta = np.array(df_disp['disp_vtheta'])
    sig_vphi = np.array(df_disp['disp_vphi'])
    df_disp['pos'] = pos
    df_disp['beta'] = 1-(sig_vtheta**2.+sig_vphi**2.)/(2*(sig_vr**2.))
    return(df_disp)


def calc_disp_beta_selectbins(data,Bins):
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    r = r[r>0.001]
    

   # print('rmin: '+str(rmin))

    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)


    bins = Bins
    h = np.histogram(r,bins=bins)
    r_binIndex = np.digitize(data_sph[:,0], bins)
    
    dict_dataSph = {'r':data_sph[:,0],
                'theta':data_sph[:,1],
                'phi': data_sph[:,2],
                'vr': data_sph[:,3],
                'vtheta': data_sph[:,4],
                'vphi': data_sph[:,5],
                'r_binIndex': r_binIndex}
    df_dataSph   = pd.DataFrame(dict_dataSph) # Data frame with positions r and their corresponding bin
    df_pos       = df_dataSph.groupby(['r_binIndex']).agg(pos = ('r',np.mean))    
    pos          = np.array(df_pos['pos'])
    df_disp = df_dataSph.groupby(['r_binIndex']).agg(
        disp_vr = ("vr", np.std),
        disp_vtheta = ("vtheta", np.std),
        disp_vphi = ("vphi", np.std))
    sig_vr = np.array(df_disp['disp_vr'])
    sig_vtheta = np.array(df_disp['disp_vtheta'])
    sig_vphi = np.array(df_disp['disp_vphi'])
    df_disp['pos'] = pos
    df_disp['beta'] = 1-(sig_vtheta**2.+sig_vphi**2.)/(2*(sig_vr**2.))
    return(df_disp)


def calc_disp_beta_linbins(data,nbins):
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    #r = r[r>0.15]
    
    rmax = max(r)
    rmin = min(r)
    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)


    # Create r bins
    log_rmin = np.log10(rmin) #logspace
    log_rmax = np.log10(rmax) #logspace
    n =nbins
    bins = np.linspace(rmin,rmax,n)
    h = np.histogram(r,bins=bins)
    r_binIndex = np.digitize(data_sph[:,0], bins)
    
    dict_dataSph = {'r':data_sph[:,0],
                'theta':data_sph[:,1],
                'phi': data_sph[:,2],
                'vr': data_sph[:,3],
                'vtheta': data_sph[:,4],
                'vphi': data_sph[:,5],
                'r_binIndex': r_binIndex}
    df_dataSph   = pd.DataFrame(dict_dataSph) # Data frame with positions r and their corresponding bin
    df_pos       = df_dataSph.groupby(['r_binIndex']).agg(pos = ('r',np.mean))    
    pos          = np.array(df_pos['pos'])
    df_disp = df_dataSph.groupby(['r_binIndex']).agg(
        disp_vr = ("vr", np.std),
        disp_vtheta = ("vtheta", np.std),
        disp_vphi = ("vphi", np.std))
    sig_vr = np.array(df_disp['disp_vr'])
    sig_vtheta = np.array(df_disp['disp_vtheta'])
    sig_vphi = np.array(df_disp['disp_vphi'])
    df_disp['pos'] = pos
    df_disp['beta'] = 1-(sig_vtheta**2.+sig_vphi**2.)/(2*(sig_vr**2.))
    return(df_disp)

def calc_disp_beta_equalN(data,nbins):
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    #r = r[r>0.15]
    
    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)
    #r        = np.sqrt(data_cart[:,0]**2.+data_cart[:,1]**2.+data_cart[:,2]**2.)


    # Create r bins
    bins = histedges_equalN(r, nbins)
    #h = np.histogram(data_sph[:,0],bins=bins)
    r_binIndex = np.digitize(data_sph[:,0], bins)
    
    dict_dataSph = {'r':data_sph[:,0],
                'theta':data_sph[:,1],
                'phi': data_sph[:,2],
                'vr': data_sph[:,3],
                'vtheta': data_sph[:,4],
                'vphi': data_sph[:,5],
                'r_binIndex': r_binIndex}
    df_dataSph   = pd.DataFrame(dict_dataSph) # Data frame with positions r and their corresponding bin
    df_pos       = df_dataSph.groupby(['r_binIndex']).agg(pos = ('r',np.mean))    
    pos          = np.array(df_pos['pos'])
    df_disp = df_dataSph.groupby(['r_binIndex']).agg(
        disp_vr = ("vr", np.std),
        disp_vtheta = ("vtheta", np.std),
        disp_vphi = ("vphi", np.std))
    sig_vr = np.array(df_disp['disp_vr'])
    sig_vtheta = np.array(df_disp['disp_vtheta'])
    sig_vphi = np.array(df_disp['disp_vphi'])
    df_disp['pos'] = pos
    df_disp['beta'] = 1-(sig_vtheta**2.+sig_vphi**2.)/(2*(sig_vr**2.))
    return(df_disp)


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))



def robust_mean_std(data, nbins):

    # Data
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    
    # Convert to spherical polar coordinates
    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)
    
    # Define r bins
    rmax = max(r)
    rmin = min(r)
    log_rmin = np.log10(rmin) #logspace
    log_rmax = np.log10(rmax) #logspace
    #n = nbins
    bins = np.logspace(log_rmin,log_rmax,nbins)
    #bins = np.linspace(2.,rmax,nbins)
    
    
    # Define variables
# =============================================================================
#     Rmin                  = 1.
#     Rmax                  = 150.
#     Redges                = np.array([Rmin,7.,7.5,8.,8.5,9.0,9.5,10.,10.5,
#                                       11.,12.,13.,Rmax])
# =============================================================================
    nR                    = nbins
    
    
# =============================================================================
#     zmin                  = 0.0
#     obsR_inplane          = np.zeros(nR)
#     obsSigmaz_inplane     = np.zeros(nR)
#     obsSigmaz_err_inplane = np.zeros(nR)
#     samplesR_inplane      = np.zeros(nR)
#     samplesSigmaz_inplane = np.zeros(nR)
# =============================================================================
    
    rmeans = np.zeros(nR)
    sigmavr = np.zeros(nR)
    sigmavtheta = np.zeros(nR)
    sigmavphi = np.zeros(nR)
    eObsPolar = np.zeros(len(data))
    eObsPolar = eObsPolar + 0.00001
    
    
    def sigmaObjFunc(sigma,vel,eVel,mu):
        y = ((vel-mu)**2.)/((sigma**2.+eVel**2.)**2.) - (1./(sigma**2.+eVel**2.))
        return (np.sum(y))
    # Number of sigma for clipping
    
    for jR in range(nR-1):
        ## OBSERVATIONS
        # Define R-age box
        boxIndex = ((bins[jR]<r) & (r<bins[jR+1]))
        
        # Calculate mean velocity
        mean_vr = np.mean(data_sph[boxIndex,3])
        mean_vtheta = np.mean(data_sph[boxIndex,4])
        mean_vphi = np.mean(data_sph[boxIndex,5])
        
        
        # Calculate initial guess for velocity dispersion
# =============================================================================
#         sigma_vr = np.std(data_sph[boxIndex,3])
#         sigma_vtheta = np.std(data_sph[boxIndex,4])
#         sigma_vphi = np.std(data_sph[boxIndex,5])
# =============================================================================
        


        # Calculate mean radii of clipped velocities
        rmeans[jR] = np.mean(r[boxIndex])
        

        # Maximise likelihood to find robust measure of observed velocity dispersion
        intervalMin = 0.
        intervalMax = 500.
        
        # Sigma vr
        optSigma_vr= scipy.optimize.brentq(sigmaObjFunc,intervalMin,intervalMax,
                                 args=(data_sph[boxIndex,3],eObsPolar[boxIndex],mean_vr),
                                 xtol=2e-12, rtol=8.881784197001252e-16,
                                 maxiter=100, full_output=False)
        sigmavr[jR] = optSigma_vr
        
        # Sigma vtheta
        optSigma_vtheta= scipy.optimize.brentq(sigmaObjFunc,intervalMin,intervalMax,
                                 args=(data_sph[boxIndex,4],eObsPolar[boxIndex],mean_vtheta),
                                 xtol=2e-12, rtol=8.881784197001252e-16,
                                 maxiter=100, full_output=False)
        sigmavtheta[jR] = optSigma_vtheta
        
        # Sigma vtheta
        optSigma_vphi= scipy.optimize.brentq(sigmaObjFunc,intervalMin,intervalMax,
                                 args=(data_sph[boxIndex,5],eObsPolar[boxIndex],mean_vphi),
                                 xtol=2e-12, rtol=8.881784197001252e-16,
                                 maxiter=100, full_output=False)
        sigmavphi[jR] = optSigma_vphi
        
        
        
    return (rmeans,sigmavphi,sigmavtheta,sigmavr)



def calc_mean_v(data,nbins):
    """
    Calculate average v component through spherically averaging @r through binning.
    
    """
    x  = data[:,0]
    y  = data[:,1]
    z  = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    r = np.sqrt(x**2.+y**2.+z**2.)
    r = r[r>0.001]
    
    rmax = max(r)
    rmin = min(r)
   # print('rmin: '+str(rmin))

    
    data_cart = np.column_stack((x,y,z,vx,vy,vz))
    data_sph = ct.CartesianToSpherical(data_cart)


    # Create r bins
    log_rmin = np.log10(rmin) #logspace
    log_rmax = np.log10(rmax) #logspace
    n =nbins
    bins = np.logspace(log_rmin,log_rmax,n)
    h = np.histogram(r,bins=bins)
    r_binIndex = np.digitize(data_sph[:,0], bins)
    
    dict_dataSph = {'r':data_sph[:,0],
                'theta':data_sph[:,1],
                'phi': data_sph[:,2],
                'vr': data_sph[:,3],
                'vtheta': data_sph[:,4],
                'vphi': data_sph[:,5],
                'r_binIndex': r_binIndex}
    df_dataSph   = pd.DataFrame(dict_dataSph) # Data frame with positions r and their corresponding bin
    df_pos       = df_dataSph.groupby(['r_binIndex']).agg(pos = ('r',np.mean))    
    pos          = np.array(df_pos['pos'])
    df_disp = df_dataSph.groupby(['r_binIndex']).agg(
        mean_vr = ("vr", np.mean),
        mean_vtheta = ("vtheta", np.mean),
        mean_vphi = ("vphi", np.mean))
    df_disp['pos'] = pos

    return(df_disp)










