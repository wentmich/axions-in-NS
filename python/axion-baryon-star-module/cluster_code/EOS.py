import numpy as np
import scipy.interpolate as interpol
from scipy.interpolate import UnivariateSpline
from constants_GeV import *

# IMPORT THE EOS FUNCTION 
DIRECTORY = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/SLy4/"
EOSdata  = np.loadtxt(DIRECTORY + "sly4_rho_p_GeV.csv", dtype=float, delimiter=",")
dEOSdata = np.loadtxt(DIRECTORY + "sly4_rho_c2_GeV.csv", dtype=float, delimiter=",")
NRHOdata = np.loadtxt(DIRECTORY + "sly4_n_rho_GeV.csv", dtype=float, delimiter=",")
NMUNdata = np.loadtxt(DIRECTORY + "sly4_n_muN_GeV.csv", dtype=float, delimiter=",")
density_low_domain = EOSdata[0, 0]
domainvals, rangevals = EOSdata[:, 0], EOSdata[:, 1]
drangevals = dEOSdata[:, 1]
ndomainvals, nrangevals = NRHOdata[:, 0], NRHOdata[:, 1]
mudomainvals, murangevals = NRHOdata[:, 0], (EOSdata[:, 0] + EOSdata[:, 1]) / NRHOdata[:, 0]

EOS_int = UnivariateSpline(domainvals, rangevals, k=4, s=0)
dEOSdrho_int = UnivariateSpline(domainvals, drangevals, k=4, s=0)
nrho_int = UnivariateSpline(ndomainvals, nrangevals, k=4, s=0)
nmun_int = UnivariateSpline(mudomainvals, murangevals, k=4, s=0)

def EOS(rho):
    return np.piecewise(rho, [rho >= density_low_domain, rho < density_low_domain], [lambda var: EOS_int(var), 0.0])

def dEOSdrho(rho):
    return np.piecewise(rho, [rho >= density_low_domain, rho < density_low_domain], [lambda var: dEOSdrho_int(var), dEOSdrho_int(density_low_domain)])

def NRHO(nbaryon):
    return np.piecewise(nbaryon, [nbaryon >= ndomainvals[0], nbaryon < ndomainvals[0]], [lambda var: nrho_int(var), lambda var: var*mN])

##### STILL NEED TO FIX CHEMICAL POTENTIAL TO BE THE ACTUAL BARYON CHEMICAL POTENTIAL
def NMUN(nbaryon):
    return np.piecewise(nbaryon, [nbaryon >= ndomainvals[0], nbaryon < ndomainvals[0]], [lambda var: nmun_int(var), mN])

