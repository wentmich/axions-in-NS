import numpy as np
from numpy import cos, sin, tan, log
from constants_GeV import *
from EOS import *
from TOV_initial_state import *
from metric_solver_radial_RK4 import *
from matter_and_metric_solver import *
from gravitational_observables import *
from density_pressure_functions import *
from save_data import *




def get_total_mass(rvals, F0):
    return -rvals[-1] * (F0[-1]**2 - 1) / (2*GNnat)

def get_NS_radius(rvals, nN0, nNCUT):
    i = 0
    while nN0[i] > nNCUT and i < len(rvals):
        i = i + 1
    
    NSRadius = rvals[i]
    return NSRadius

def get_NS_mass_and_radius(rvals, nN0, nNCUT, F0):
    i = 0
    while nN0[i] > nNCUT and i < len(rvals):
        i = i + 1
    
    NSRadius = rvals[i]
    
    NSMass = -rvals[i] * (F0[i]**2 - 1) / (2*GNnat)
    
    return [NSRadius, NSMass];

def get_NS_mass_and_radius_from_G0(rvals, nN0, nNCUT, G0):
    i = 0
    while nN0[i] > nNCUT and i < len(rvals):
        i = i + 1
    
    NSRadius = rvals[i]
    
    NSMass = rvals[i] * (1 - G0[i]**(-2)) / (2*GNnat)
    
    return [NSRadius, NSMass, i];

def axion_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    axion_density_vals = axion_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    Mavals = np.zeros(np.shape(rvals))
    for i in range(len(rvals)):
        Mavals[i] = np.sum(axion_density_vals[:i]*4*np.pi*rvals[:i]**2)*dr
    return Mavals;

def baryon_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    baryon_density_vals = baryon_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    Mvals = np.zeros(np.shape(rvals))
    for i in range(len(rvals)):
        Mvals[i] = np.sum(baryon_density_vals[:i]*4*np.pi*rvals[:i]**2)*dr
    return Mvals;

def axion_mass_total(rvals, G0, nN0, U0, A0, P0, epsilon, fa, RNS):
    Mavals = axion_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    i = 0
    while rvals[i] <= RNS:
        i += 1
    return Mavals[i];

def baryon_mass_total(rvals, G0, nN0, U0, A0, P0, epsilon, fa, RNS):
    Mvals = baryon_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    i = 0
    while rvals[i] <= RNS:
        i += 1
    return Mvals[i];

# tidal deformability calculator
def ky0_fnc(r, Y00, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0):
    ky0 = pow(r,-1)*pow(-2*GNnat*(Ma0 + MN0) + r,-2)*(-4*pow(GNnat,2)*pow(Ma0 + MN0,3)*(-1 + pow(Y00,2)) + 2*GNnat*r*pow(Ma0 + MN0,2)*(-6 + Y00 + 4*GNnat*Pi*EOS(NRHO(nN0))*pow(r,2)*(15 + Y00 - 2*pow(Y00,2)) + 2*(10*GNnat*Pi*Ra0*pow(r,2) - 8*GNnat*Pat0*Pi*(-3 + Y00)*pow(r,2) - 2*GNnat*Pi*Ra0*Y00*pow(r,2) + 2*GNnat*Pa0*Pi*(3 + (5 - 2*Y00)*Y00)*pow(r,2) + 10*GNnat*Pi*NRHO(nN0)*pow(r,2) - 2*GNnat*Pi*Y00*NRHO(nN0)*pow(r,2) + 4*dRadr0*GNnat*Pi*pow(r,3) + 4*dnNdr*GNnat*Pi*NMUN(nN0)*pow(r,3) + pow(Y00,2))) + (Ma0 + MN0)*pow(r,2)*(6 - Y00 - 64*GNnat*Pat0*Pi*pow(r,2) - 20*GNnat*Pi*Ra0*pow(r,2) + 32*GNnat*Pat0*Pi*Y00*pow(r,2) + 4*GNnat*Pi*Ra0*Y00*pow(r,2) - 20*GNnat*Pi*NRHO(nN0)*pow(r,2) + 4*GNnat*Pi*Y00*NRHO(nN0)*pow(r,2) + 4*GNnat*Pa0*Pi*pow(r,2)*(-5 + Y00*(-7 + 4*Y00) + 8*GNnat*Pi*(8*Pat0 - (-5 + Y00)*(Ra0 + NRHO(nN0)))*pow(r,2)) - 16*dRadr0*GNnat*Pi*pow(r,3) - 16*dnNdr*GNnat*Pi*NMUN(nN0)*pow(r,3) + 32*(7 + Y00)*pow(GNnat,2)*pow(Pa0,2)*pow(Pi,2)*pow(r,4) - pow(Y00,2) + 4*GNnat*Pi*EOS(NRHO(nN0))*pow(r,2)*(-21 + Y00 + 16*GNnat*Pa0*Pi*(11 + Y00)*pow(r,2) + 8*GNnat*Pi*(8*Pat0 - (-5 + Y00)*(Ra0 + NRHO(nN0)))*pow(r,2) + 4*pow(Y00,2)) + 32*(15 + Y00)*pow(GNnat,2)*pow(Pi,2)*pow(r,4)*pow(EOS(NRHO(nN0)),2)) + 4*Pi*pow(r,5)*(-2*Pat0*(-1 + Y00) + r*(dRadr0 + dnNdr*NMUN(nN0)) - 4*GNnat*Pi*(1 + Y00)*pow(Pa0,2)*pow(r,2) + 64*pow(GNnat,2)*pow(Pa0,3)*pow(Pi,2)*pow(r,4) + EOS(NRHO(nN0))*(-((-2 + Y00)*(3 + Y00)) - 8*GNnat*Pa0*Pi*(5 + Y00)*pow(r,2) + 4*GNnat*Pi*(-8*Pat0 + (-5 + Y00)*(Ra0 + NRHO(nN0)))*pow(r,2) + 192*pow(GNnat,2)*pow(Pa0,2)*pow(Pi,2)*pow(r,4)) + Pa0*(4 + Y00 + 4*GNnat*Pi*(-8*Pat0 + (-5 + Y00)*(Ra0 + NRHO(nN0)))*pow(r,2) - pow(Y00,2)) + 4*GNnat*Pi*pow(r,2)*(-9 - Y00 + 48*GNnat*Pa0*Pi*pow(r,2))*pow(EOS(NRHO(nN0)),2) + 64*pow(GNnat,2)*pow(Pi,2)*pow(r,4)*pow(EOS(NRHO(nN0)),3)))*pow(Ma0 + MN0 + 4*Pi*(Pa0 + EOS(NRHO(nN0)))*pow(r,3),-1)
    return ky0;

def solve_for_y0_next_step(r, Y00, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0, dr):
    # get k1
    k1yr = ky0_fnc(r, Y00, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0)
    # get k2
    k2yr = ky0_fnc(r + dr/2, Y00 + k1yr*dr/2, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0)
    # get k3
    k3yr = ky0_fnc(r + dr/2, Y00 + k2yr*dr/2, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0)
    # get k4
    k4yr = ky0_fnc(r + dr, Y00 + k3yr*dr, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0)
    # update with Runge-Kutta
    Y01 = Y00 + (1/6)*k1yr*dr + (1/3)*k2yr*dr + (1/3)*k3yr*dr + (1/6)*k4yr*dr
    return np.array(Y01).astype(np.float64);

def solve_for_y0(rvals, MN0, Ma0, nN0, dnNdr, Ra0, Pa0, Pat0, dRadr0):
    dr = rvals[1] - rvals[0]
    y0initial = 2
    y0vals = np.zeros(np.shape(rvals))
    Nr = np.shape(rvals)[0]
    y0vals[0] = y0initial
    i = 1
    while nN0[i-1] > 0.0 and i < Nr:
        y0vals[i] = solve_for_y0_next_step(rvals[i-1], y0vals[i-1], MN0[i-1], Ma0[i-1], nN0[i-1], dnNdr[i-1], Ra0[i-1], Pa0[i-1], Pat0[i-1], dRadr0[i-1], dr)
        i += 1
    
    return y0vals;

def Lambda_total(Rs, Y00, MN0, Ma0):
    return ((16*(-(Rs*(-2 + Y00)) + 2*GNnat*(Ma0 + MN0)*(-1 + Y00))*pow(GNnat,5)*pow(Rs,2)*pow(-2*GNnat*(Ma0 + MN0) + Rs,2)*pow(2*GNnat*(Ma0 + MN0)*(2*Rs*(-2 + 3*Y00)*pow(GNnat,3)*pow(Ma0 + MN0,3) + 4*(1 + Y00)*pow(GNnat,4)*pow(Ma0 + MN0,4) + 2*(13 - 11*Y00)*pow(GNnat,2)*pow(Ma0 + MN0,2)*pow(Rs,2) + 3*GNnat*(Ma0 + MN0)*(-8 + 5*Y00)*pow(Rs,3) - 3*(-2 + Y00)*pow(Rs,4)) - 3*(Rs*(-2 + Y00) - 2*GNnat*(Ma0 + MN0)*(-1 + Y00))*log(1 - 2*GNnat*(Ma0 + MN0)*pow(Rs,-1))*pow(Rs,2)*pow(-2*GNnat*(Ma0 + MN0) + Rs,2),-1))/15.) / (GNnat**5)

def solve_tidal_deformability(rvals, G0, nN0, U0, A0, P0, epsilon, fa, nNCUT):
    dr = rvals[1] - rvals[0]

    # first get the index at which the radius is located and get the radius
    radius_and_index = get_NS_mass_and_radius_from_G0(rvals, nN0, nNCUT, G0)
    Rstar = radius_and_index[0]
    Mtotal = radius_and_index[1]
    rindex = radius_and_index[2]
    CNS = Mtotal / Rstar

    MN0vals = baryon_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    Ma0vals = axion_mass_function(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    dnNdrvals = first_r_derivative(nN0, dr)
    Ra0vals = axion_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    Pa0vals = axion_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    Pat0vals = axion_angular_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa)
    dRadr0vals = first_r_derivative(Ra0vals, dr)

    # solve y0 and y1 ODEs and assign values to y0and y1 at Rstar
    y0vals = solve_for_y0(rvals, MN0vals, Ma0vals, nN0, dnNdrvals, Ra0vals, Pa0vals, Pat0vals, dRadr0vals)
    y0star = y0vals[rindex]
    
    # solve for each contribution to tidal deformability
    LambdaTotal = Lambda_total(Rstar, y0star, MN0vals[rindex], Ma0vals[rindex])
    
    return np.array([CNS, LambdaTotal]);


def calculate_gravitational_observables(rvals, Gv, nNv, Uv, Av, Pv, epsilon, fa, nNRadiusCUT):
    # get the NS mass and radius total
    RMtotvals = get_NS_mass_and_radius_from_G0(rvals, nNv, nNRadiusCUT, Gv)
    RNS = RMtotvals[0]
    MNS = RMtotvals[1]

    # get tidal deformability
    LCvals = solve_tidal_deformability(rvals, Gv, nNv, Uv, Av, Pv, epsilon, fa, nNRadiusCUT)
    CNS = LCvals[0]
    LNS = LCvals[1]

    return np.array([RNS, MNS, CNS, LNS]);
