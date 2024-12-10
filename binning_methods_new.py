#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:45:58 2023

@author: paulagherghinescu
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
import agama
from multiprocessing import Pool
import pickle
import astropy
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.io import ascii
from astropy.table import Table
import CoordTrans as ct


agama.setUnits(mass=1,length=1,velocity=1)

#%%

def dens_data_binning(x,y,z,mass,nbins):

    # Sample data: Replace this with your dataset
    masses = mass
    distances = np.sqrt(x**2.+y**2.+z**2.)
    
    # Define parameters for binning
    num_bins =nbins
    max_radius = distances.max()
    min_radius = distances.min()
    bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), num_bins + 1)
    
    # Initialize arrays to store densities and radii
    densities = np.zeros(num_bins)
    density_errors = np.zeros(num_bins)
    radii = np.zeros(num_bins)
    
    # Calculate density and error for each bin
    for i in range(num_bins):
        # Find particles within the current bin
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        particles_in_bin = masses[mask]
        nr_part_bin = np.sum(mask)
        print(nr_part_bin)
    
        # Calculate density within the bin
        volume = (4/3) * np.pi * (bin_edges[i + 1]**3 - bin_edges[i]**3)
        density = np.sum(particles_in_bin) / volume
    
        # Calculate the standard error using Poisson statistics
        error = density/(np.sqrt(nr_part_bin/2))
        #error *= np.log(10)
    
        # Store density and error
        densities[i] = density
        density_errors[i] = error
        radii[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        
    return (radii,densities,density_errors)


def vel_dispersion_cartesian_error(data,nbins, err):

    # Sample data: Replace this with your dataset
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    vz = data[:,5]
    
    distances = np.sqrt(x**2.+y**2.+z**2.)
    
    # Define parameters for binning
    num_bins =nbins
    max_radius = distances.max()
    min_radius = distances.min()
    #bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), num_bins + 1)
    bin_edges = np.logspace(np.log10(0.1), np.log10(200), num_bins + 1)
    
    # Initialize arrays to store velocity dispersions and radii
    sigvx = np.zeros(num_bins)
    sigvy = np.zeros(num_bins)
    sigvz = np.zeros(num_bins)
    sigvx_err = np.zeros(num_bins)
    sigvy_err = np.zeros(num_bins)
    sigvz_err = np.zeros(num_bins)
    radii = np.zeros(num_bins)
    
    # Calculate density and error for each bin
    for i in range(num_bins):
        # Find particles within the current bin
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        vx_in_bin = vx[mask]
        vy_in_bin = vy[mask]
        vz_in_bin = vz[mask]
        
        nr_part_bin = np.sum(mask)
        #print(nr_part_bin)
    
        # Calculate density within the bin
        sigvx_ = np.std(vx_in_bin)
        sigvy_ = np.std(vy_in_bin)
        sigvz_ = np.std(vz_in_bin)
    
        # Calculate the standard error using Poisson statistics
        sigvx_err_ = sigvx_/(np.sqrt(2*nr_part_bin-1)) # not sure this formula is correct???
        sigvy_err_ = sigvy_/(np.sqrt(2*nr_part_bin-1))
        sigvz_err_ = sigvz_/(np.sqrt(2*nr_part_bin-1))
        
    
        # Store density and error
        sigvx[i] = sigvx_
        sigvy[i] = sigvy_
        sigvz[i] = sigvz_
        
        sigvx_err[i] = sigvx_err_
        sigvy_err[i] = sigvy_err_
        sigvz_err[i] = sigvz_err_

        radii[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
    
    if (err):
        return (radii,sigvx,sigvy,sigvz,sigvx_err,sigvy_err,sigvz_err)
    else:
        return (radii,sigvx,sigvy,sigvz)
    
def vel_dispersion_cylindrical(data,nbins):

    # Sample data: Replace this with your dataset
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    
    distances = np.sqrt(x**2.+y**2.+z**2.)
    
    data_cyl = ct.CartesianToPolar(data)
    
    R = data_cyl[:,0]
    phi = data_cyl[:,1]
    z = data_cyl[:,2]
    vR = data_cyl[:,3]
    vphi = data_cyl[:,4]
    vz = data_cyl[:,5]
    
    # Define parameters for binning
    num_bins =nbins
    max_radius = distances.max()
    min_radius = distances.min()
    #bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), num_bins + 1)
    bin_edges = np.logspace(np.log10(0.1), np.log10(200), num_bins + 1)
    
    # Initialize arrays to store velocity dispersions and radii
    sigvR = np.zeros(num_bins)
    sigvphi = np.zeros(num_bins)
    sigvz = np.zeros(num_bins)
    radii = np.zeros(num_bins)
    
    # Calculate density and error for each bin
    for i in range(num_bins):
        # Find particles within the current bin
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        vR_in_bin = vR[mask]
        vphi_in_bin = vphi[mask]
        vz_in_bin = vz[mask]
        
        #nr_part_bin = np.sum(mask)
        #print(nr_part_bin)
    
        # Calculate vel dispersion within the bin
        sigvR_ = np.std(vR_in_bin)
        sigvphi_ = np.std(vphi_in_bin)
        sigvz_ = np.std(vz_in_bin)
    
        sigvR[i] = sigvR_
        sigvphi[i] = sigvphi_
        sigvz[i] = sigvz_

        radii[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
    
    return (radii,sigvR,sigvphi,sigvz)
        
    
def vel_dispersion_cartesian_binning(stars,nbins,log_bins):  
    
    x = stars[:,0]
    y = stars[:,1]
    z = stars[:,2]
    vx = stars[:,3]
    vy = stars[:,4]
    vz = stars[:,5]
    
    # Calculate the radius for each star
    radius = np.sqrt(x**2 + y**2 + z**2)
    
    # Divide the stars into radial bins
    if (log_bins):
        bins = np.logspace(np.log10(min(radius)), np.log10(max(radius)), nbins + 1)
    else:
        bins = np.linspace(0, max(radius), nbins + 1)
        
    digitized = np.digitize(radius, bins)
    
    # Calculate velocity dispersion for each bin
    velocity_dispersion = []
    sig_vx  = []
    sig_vy  = []
    sig_vz  = []
    bin_centers = []
    
    for i in range(1, nbins + 1):
        in_bin = digitized == i
        if np.any(in_bin):
            bin_centers.append(np.mean(radius[in_bin]))
            dispersion = np.std(np.sqrt(vx[in_bin]**2 + vy[in_bin]**2 + vz[in_bin]**2))
            sig_vx_ = np.std(vx[in_bin])
            sig_vy_ = np.std(vy[in_bin])
            sig_vz_ = np.std(vz[in_bin])
            velocity_dispersion.append(dispersion)
            sig_vx.append(sig_vx_)
            sig_vy.append(sig_vy_)
            sig_vz.append(sig_vz_)
            
    velocity_dispersion = np.array(velocity_dispersion)
    sig_vx = np.array(sig_vx)
    sig_vy = np.array(sig_vy)
    sig_vz = np.array(sig_vz)
    
    return (bin_centers, velocity_dispersion, sig_vx,sig_vy,sig_vz)


def vel_dispersion_spherical_binning(stars,nbins,log_bins):  
    
    # Convert
    stars_sph = ct.CartesianToSpherical(stars)
    r = stars_sph[:,0]
    theta = stars_sph[:,1]
    phi = stars_sph[:,2]
    vr  = stars_sph[:,3]
    vt = stars_sph[:,4]
    vp = stars_sph[:,5]
    
    # Calculate the radius for each star
    radius = np.copy(r)
    
    # Divide the stars into radial bins
    if (log_bins):
        bins = np.logspace(np.log10(min(radius)), np.log10(max(radius)), nbins + 1)
    else:
        bins = np.linspace(min(radius), max(radius), nbins + 1)
        
    
    digitized = np.digitize(radius, bins)
    
    # Calculate velocity dispersion for each bin
    sig_vr  = []
    sig_vt  = []
    sig_vp  = []
    bin_centers = []
    
    for i in range(1, nbins + 1):
        in_bin = digitized == i
        if np.any(in_bin):
            bin_centers.append(np.mean(radius[in_bin]))
            sig_vr_ = np.std(vr[in_bin])
            sig_vt_ = np.std(vt[in_bin])
            sig_vp_ = np.std(vp[in_bin])
            sig_vr.append(sig_vr_)
            sig_vt.append(sig_vt_)
            sig_vp.append(sig_vp_)
            
    sig_vr = np.array(sig_vr)
    sig_vt = np.array(sig_vt)
    sig_vp = np.array(sig_vp)
    
    return (bin_centers, sig_vr,sig_vt,sig_vp)


def number_desnity_binning(stars,nbins):
    x = stars[:,0]
    y = stars[:,1]
    z = stars[:,2]
    
    # Calculate the radius for each star
    radius = np.sqrt(x**2 + y**2 + z**2)
    
    # Divide the stars into radial bins
    bins = np.logspace(np.log10(min(radius)), np.log10(max(radius)), nbins + 1)
    digitized = np.digitize(radius, bins)
    
    # Calculate number density for each bin
    number_density = []
    bin_centers = []
    
    for i in range(1, nbins + 1):
        in_bin = digitized == i
        if np.any(in_bin):
            bin_centers.append(np.mean(radius[in_bin]))
            density = np.sum(in_bin) / (4/3 * np.pi * (bins[i]**3 - bins[i-1]**3))
            number_density.append(density)
    
    return bin_centers, number_density

def truncate_data(stars,rmax):
    # Calculate the radius for each star
    x  = stars[:,0]
    y  = stars[:,1]
    z  = stars[:,2]
    vx = stars[:,3]
    vy = stars[:,4]
    vz = stars[:,5]
    
    radius = np.sqrt(x**2 + y**2 + z**2)

    # Mask for stars with radius below rmax
    mask = (radius <= rmax)

    # Filter stars based on the mask
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    vx_filtered = vx[mask]
    vy_filtered = vy[mask]
    vz_filtered = vz[mask]
    
    stars_filtered = np.column_stack((x_filtered,y_filtered,z_filtered,vx_filtered,vy_filtered,vz_filtered))
    
    return (stars_filtered)


