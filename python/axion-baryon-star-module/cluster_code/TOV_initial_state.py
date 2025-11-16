# expansion values near r=0
from EOS import *
import numpy as np
from numpy import cos, sin, tan, roll
from constants_GeV import *
from auxilliary_functions import *




def nN2_fnc_TOV(n0):
    p0 = EOS(NRHO(n0))
    return -0.5*(pow(dEOSdrho(NRHO(n0)),-1)*pow(NMUN(n0),-1)*((16*GNnat*p0*Pi*NRHO(n0))/3. + 4*GNnat*Pi*pow(p0,2) + (4*GNnat*Pi*pow(NRHO(n0),2))/3.))

def nN3_fnc_TOV(n0):
    return 0.0

def G0_fnc_TOV(n0):
    return 1.0

def G2_fnc_TOV(n0):
    return (4*GNnat*Pi*NRHO(n0))/3.

def G3_fnc_TOV(n0):
    return 0.0

# spatial derivatives
def kGr_fnc_TOV(r, nN0, G0):
    return (pow(r,-1)*(G0 + pow(G0,3)*(-1 + 8*GNnat*Pi*NRHO(nN0)*pow(r,2))))/2.

def knNr_fnc_TOV(r, nN0, G0):
    return pow(r,-1)*pow(dEOSdrho(NRHO(nN0)),-1)*(NRHO(nN0)*(0.5 - 0.5*pow(G0,2)) + EOS(NRHO(nN0))*(0.5 + pow(G0,2)*(-0.5 - 8.419468311620646e-38*NRHO(nN0)*pow(r,2))) - 8.419468311620646e-38*pow(G0,2)*pow(r,2)*pow(EOS(NRHO(nN0)),2))*pow(NMUN(nN0),-1)

def kFr_fnc_TOV(r, F0, nN0, G0):
    return F0*pow(r,-1)*(-0.5 + pow(G0,2)*(0.5 + 8.419468311620646e-38*EOS(NRHO(nN0))*pow(r,2)))

# solver
def solve_for_G_N_noaxion_next_step(r, G0, nN0, dr):
    # get k1
    k1Gr  = kGr_fnc_TOV(r, nN0, G0)
    k1nNr = knNr_fnc_TOV(r, nN0, G0)
    # get k2
    k2Gr  = kGr_fnc_TOV(r + dr/2, nN0 + k1nNr*dr/2, G0 + k1Gr*dr/2)
    k2nNr = knNr_fnc_TOV(r + dr/2, nN0 + k1nNr*dr/2, G0 + k1Gr*dr/2)
    # get k3
    k3Gr  = kGr_fnc_TOV(r + dr/2, nN0 + k2nNr*dr/2, G0 + k2Gr*dr/2)
    k3nNr = knNr_fnc_TOV(r + dr/2, nN0 + k2nNr*dr/2, G0 + k2Gr*dr/2)
    # get k4
    k4Gr  = kGr_fnc_TOV(r + dr, nN0 + k3nNr*dr, G0 + k3Gr*dr)
    k4nNr = knNr_fnc_TOV(r + dr, nN0 + k3nNr*dr, G0 + k3Gr*dr)
    # update with Runge-Kutta
    G1  = G0  + (1/6)*k1Gr*dr  + (1/3)*k2Gr*dr  + (1/3)*k3Gr*dr  + (1/6)*k4Gr*dr
    nN1 = nN0 + (1/6)*k1nNr*dr + (1/3)*k2nNr*dr + (1/3)*k3nNr*dr + (1/6)*k4nNr*dr
    return np.array([G1, nN1]).astype(np.float64);

def solve_for_G_N_noaxion(nNinitial, nNCUT, rvals):
    dr = rvals[1] - rvals[0]
    Nr = len(rvals)
    Ginitial  = G0_fnc_TOV(nNinitial) + G2_fnc_TOV(nNinitial)*rvals[0]**2 + G3_fnc_TOV(nNinitial)*rvals[0]**3
    nNinitial = nNinitial + nN2_fnc_TOV(nNinitial)*rvals[0]**2 + nN3_fnc_TOV(nNinitial)*rvals[0]**3
    
    G1vals  = np.zeros(np.shape(rvals))
    nN1vals = np.zeros(np.shape(rvals))
    
    G1vals[0]  = Ginitial
    nN1vals[0] = nNinitial
    
    i = 1
    while nN1vals[i-1] > nNCUT and i < Nr:
        newvals = solve_for_G_N_noaxion_next_step(rvals[i-1], G1vals[i-1], nN1vals[i-1], dr)
        G1vals[i]  = newvals[0]
        nN1vals[i] = newvals[1]
        i += 1
    
    i = i - 1
    r0 = rvals[i]
    n0 = nN1vals[i]
    while i < Nr:
        newvals = solve_for_G_N_noaxion_next_step(rvals[i-1], G1vals[i-1], nN1vals[i-1], dr)
        G1vals[i]  = newvals[0]
        nN1vals[i] = 0.0#n0 * np.exp(-(rvals[i] - r0)/r0)
        i += 1
        
    return np.concatenate((G1vals, nN1vals)).astype(np.float64);
