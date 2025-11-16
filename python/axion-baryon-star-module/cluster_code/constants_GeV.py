import numpy as np

Msun      = 1.989*10**30
GNnat     = 6.7e-39
clight    = 2.99792458*10**8
mN        = 0.929532
hbarGeV   = 6.582119569*10**(-25)
sigmaN   = 0.059
mpi       = 0.134977
fpi       = 0.130;
mu        = 0.0017;
md        = 0.0041;
beta      = np.sqrt(4 * mu*md / ((mu+md)**2))
E         = np.e
Pi        = np.pi
nsat_in_perfm3 = 0.159
perfm3_2_perm3 = (10**(-15))**(-3)
fm_2_perGeV = 10**(-15) / (clight * hbarGeV)
m_2_perGeV = 1 / (clight * hbarGeV)
perfm3_2_GeV3 = 1 / (fm_2_perGeV**3)
perm3_2_GeV3 = 1 / (m_2_perGeV**3)
nsatinperm3 = nsat_in_perfm3 * perfm3_2_perm3
nsatinGeV3  = nsat_in_perfm3 * perfm3_2_GeV3
GeV_2_kg = 1.78266e-27
kg_2_Msun = 1 / Msun
GeV_2_km = 1 / 5.06773e+18
DENCUT = 1e-5*nsatinGeV3

# sites and scale doubles for halving dr
NFRICTIONSITES = 2000
FRICTIONSITESCALE = 2000
ZETAFRICTION = 4

INTERMEDIATEMETRICINTEGRATOR = 0
VISCOSITYORDER = 2
