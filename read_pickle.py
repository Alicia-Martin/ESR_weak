import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import math

import numpy as np
from scipy.optimize import curve_fit
from colossus.halo import profile_nfw
from colossus.cosmology import cosmology
import scipy.optimize
from esr.esd import ExcessSurfaceDensity
import numdifftools as nd

import jax.numpy as jnp
import jax

from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import numpyro



SNRs = []
clusters = []

def esd_nfw(M, c, z, xvar, yvar, yerr):
    h = 0.7
    # Create an NFW profile
    nfw = profile_nfw.NFWProfile(M=M, c=c, z=z, mdef='200c')
    x = xvar*10**3*0.7 # Convert Mpc to kpc/h
    esd = nfw.deltaSigma(x)
    esd = esd*h/10**6 #Convert M0*h/kpc^2 to 10^12*M0/Mpc^2

    esd2 = ExcessSurfaceDensity.calculate(x, nfw.density)
    esd2 = esd2*h/10**6 #Convert M0*h/kpc^2 to 10^12 M0/Mpc^2

    plt.plot(xvar,esd2,ls='--')
    # plt.plot(xvar, esd)
    plt.errorbar(xvar, yvar, yerr, fmt='o', label='Data')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    return esd2

def model(xvar, yvar=None, yerr=None, z=0.186):
    cosmo = cosmology.setCosmology('planck15')
    h = cosmo.h
    def get_esd(M, c, z, xvar):
        nfw = profile_nfw.NFWProfile(M=M, c=c, z=z, mdef='200c')
        x = xvar*10**3*h # Convert Mpc to kpc/h
        esd = ExcessSurfaceDensity.calculate(x, nfw.density)
        esd = esd*h/(10**6) #Convert M0*h/kpc^2 to 10^12 M0/Mpc^2
        return esd
    # Priors for the mass and concentration parameters
    M = numpyro.sample("M", dist.Uniform(1e14, 1e16))
    c = numpyro.sample("c", dist.Uniform(1.0, 10.0))
    
    # Expected ESD values
    esd_model = get_esd(M, c, z, xvar)
    
    # Likelihood
    with numpyro.plate("data", len(xvar)):
        numpyro.sample("obs", dist.Normal(esd_model, yerr), obs=yvar)

def run_mcmc(xvar, yvar, yerr):
    # Define the MCMC method
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=2)
    
    # Run MCMC
    mcmc.run(jax.random.PRNGKey(0), xvar=xvar, yvar=yvar, yerr=yerr)
    
    # Get results
    mcmc.print_summary()
    samples = mcmc.get_samples()
    
    M_fit = jnp.median(samples["M"])
    c_fit = jnp.median(samples["c"])
    
    return M_fit, c_fit, samples

def fit_nfw(xvar, yvar, yerr):    
    def get_esd(M, c, z, xvar):
        nfw = profile_nfw.NFWProfile(M=M, c=c, z=z, mdef='200c')
        x = xvar*10**3*h # Convert Mpc to kpc/h
        esd = ExcessSurfaceDensity.calculate(x, nfw.density)
        esd = esd*h/(10**6) #Convert M0*h/kpc^2 to 10^12 M0/Mpc^2
        return esd
         

    def likelihood(params , xvar, yvar, yerr, z):
        M, c = params
        # print(M, c)
        esd_nfw = get_esd(M, c, z, xvar)
        return jnp.sum((yvar - esd_nfw)**2 / (2*yerr)**2)
    
    cosmo = cosmology.setCosmology('planck15')
    h = cosmo.h
    initial_guess = [5e14, 5] 
    z = 0.186
    sol = scipy.optimize.minimize(likelihood, initial_guess, args=(xvar, yvar, yerr, z), method='Nelder-Mead') 
    M_fit, c_fit = sol.x
    print(M_fit/10**14/h, c_fit)

    #Calculate errors
    def fop(x):
        return likelihood(x, xvar, yvar, yerr, z)
    Hfun = nd.Hessian(fop)
    Hmat = Hfun(sol.x)
    Fisher_diag = jnp.diag(Hmat)
    Delta = np.sqrt(12./Fisher_diag)

    # print(Delta)


    plt.plot(xvar, get_esd(M_fit, c_fit, z, xvar), label='Fit')
    plt.errorbar(xvar, yvar, yerr, fmt='o', label='Data')
    plt.show()

    # def esd_model(xvar, M, c):
    #     return get_esd(M, c, z, xvar)

    # params, params_covariance = curve_fit(esd_model, xvar, yvar, sigma=yerr, p0=initial_guess, absolute_sigma=True)
    # print(params)

    return M_fit, c_fit


# Set the cosmology
cosmo = cosmology.setCosmology('planck15')

with open('esr/dark_matter_data.txt', 'r') as file:
        lines = file.readlines()

# Extract the data
xvar = np.array([float(value) for value in lines[0].strip().split(', ')])
yvar = np.array([float(value) for value in lines[1].strip().split(', ')])
yerr = np.array([float(value) for value in lines[2].strip().split(', ')])

cuts = (xvar <= 3/0.7) & (xvar >= 0.3/0.7)
xvar = xvar[cuts]
yvar = yvar[cuts]
yerr = yerr[cuts]

d = np.sum(yvar/(yerr)**2)/np.sum(1/(yerr)**2)
sigma = 1/np.sqrt(np.sum(1/yerr**2))
SNR_i = d/sigma
print(SNR_i)

# params = fit_nfw(xvar, yvar, yerr)

pickle_files = glob.glob('XXL/*.pickle')
# i = 91
# pickle_files = ['XXL/' + str(i) + '.pickle']
# pickle_files = ['91.pickle'] #Same binning as the paper

for file_path in pickle_files:
    # print(file_path)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    file_name = os.path.basename(file_path)
    cluster_name = os.path.splitext(file_name)[0]  

    cuts = (data['rp'] <= 3/0.7) & (data['rp'] >= 0.3/0.7)
    x = data['rp'][cuts]
    ESD = data['ds'][cuts]
    ESD_err = data['ds_err'][cuts]

    # params = fit_nfw(x, ESD, ESD_err)

    # Run MCMC on the data
    # M_fit, c_fit, samples = run_mcmc(xvar, yvar, yerr)

    # plt.errorbar(x, ESD, yerr=ESD_err, fmt='o', label='Data cluster 91')
    # plt.errorbar(xvar, yvar, yerr, fmt='o', label='Data before')
    # plt.legend()
    # plt.show()


    d = np.sum(ESD/(ESD_err)**2)/np.sum(1/(ESD_err)**2)
    sigma = 1/np.sqrt(np.sum(1/ESD_err**2))
    SNR_i = d/sigma

    # print(cluster_name, SNR_i)

    #Calculate SNR for the entire range
    d = np.sum(data['ds']/(data['ds_err'])**2)/np.sum(1/(data['ds_err'])**2)
    sigma = 1/np.sqrt(np.sum(1/data['ds_err']**2))
    SNR_i = d/sigma

    # print(cluster_name, SNR_i)

    SNRs.append(SNR_i)
    clusters.append(cluster_name)


    # plt.errorbar(data['rp'], data['ds'], yerr=data['ds_err'], fmt='o')
    # plt.savefig('XXL/' + cluster_name + '.png')
    # plt.close()


# Filter out NaN SNRs and corresponding clusters
filtered_SNRs_clusters = [(snr, cluster) for snr, cluster in zip(SNRs, clusters) if not math.isnan(snr)]

# Separate the SNRs and clusters again after filtering
filtered_SNRs = [snr for snr, _ in filtered_SNRs_clusters]
filtered_clusters = [cluster for _, cluster in filtered_SNRs_clusters]

# Now sort based on the absolute values of the filtered SNRs
sorted_clusters = [cluster for _, cluster in sorted(zip(filtered_SNRs, filtered_clusters), key=lambda x: abs(x[0]), reverse=True)]
# sorted_SNRs = sorted(filtered_SNRs, key=abs, reverse=True)
print(sorted_clusters)
print(len(sorted_clusters))






 