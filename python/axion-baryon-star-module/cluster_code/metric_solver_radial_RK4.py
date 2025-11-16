import numpy as np
from numpy import cos, sin, tan, roll
from auxilliary_functions import *
from constants_GeV import *
from EOS import *
from TOV_initial_state import *


# solve metric constraint equations
# first solve G constraint
def dGdr_for_RK4_G3(r, nN0, U0, A0, dAdr, P0, epsilon, fa):
    dGdronG3 = pow(r,-1)*(0.5 - 2.592333749140991e-41*epsilon*pow(r,2) - 8.419468311620646e-38*NRHO(nN0)*pow(r,2) + pow(U0,2)*(-0.5 - 8.419468311620646e-38*EOS(NRHO(nN0))*pow(r,2) + epsilon*pow(r,2)*(2.592333749140991e-41 - 1.8330567731163412e-41*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + nN0*pow(r,2)*(4.967486303856181e-39 - 3.5125432509080046e-39*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 1.8330567731163412e-41*epsilon*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))*pow(-1. + pow(U0,2),-1)
    if np.isnan(dGdronG3).any():
        print("dGdr_for_RK4_G3 is nan")
    return dGdronG3

def dGdr_for_RK4_G1(r, nN0, U0, A0, dAdr, P0, epsilon, fa):
    dGdronG1 = 4.209734155810323e-38*r*pow(dAdr,2)*pow(fa,2) + 4.209734155810323e-38*r*pow(fa,2)*pow(P0,2) + pow(r,-1)/2.
    if np.isnan(dGdronG1).any():
        print("dGdr_for_RK4_G1 is nan")
    return dGdronG1

def solve_G_next_step_RK4_fast(r, dGc1_0, dGc3_0, dGc1_half, dGc3_half, dGc1_1, dGc3_1, G0, nN0, U0, A0, P0, epsilon, fa, dr):
    k1 = dGc1_0 * G0 + dGc3_0 * G0**3
    if np.isnan(k1).any():
        print("k1 in solve_G_next_step_RK4_fast is nan")
    k2 = dGc1_half * (G0 + k1*dr/2.0) + dGc3_half * (G0 + k1*dr/2.0)**3
    if np.isnan(k2).any():
        print("k2 in solve_G_next_step_RK4_fast is nan")
    k3 = dGc1_half * (G0 + k2*dr/2.0) + dGc3_half * (G0 + k2*dr/2.0)**3
    if np.isnan(k3).any():
        print("k3 in solve_G_next_step_RK4_fast is nan")
    k4 = dGc1_1 * (G0 + k3*dr) + dGc3_1 * (G0 + k3*dr)**3
    if np.isnan(k4).any():
        print("k4 in solve_G_next_step_RK4_fast is nan") 
    ans = G0 + (k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0) * dr
    return ans

def solve_G_constraint_fast(rvals, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    G1 = np.zeros(np.shape(rvals))
    # sets initial condition at the very exterior
    G1[0] = 1.0 + (2.0*np.pi*GNnat/3) * (P0[0]**2 + 2.0*(U0[0]*EOS(NRHO(nN0[0])) + NRHO(nN0[0]))/(1 - U0[0]**2) + 2*(np.sqrt(1 - beta**2*np.sin(A0[0]/2)) * (-epsilon*mpi**2*fpi**2*(1 - U0[0]**2) + sigmaN*nN0[0]))/(1 - U0[0]**2))*rvals[0]**2
    # calculate the dGdr / G and dGdr / G^3 at different r values
    # this is maybe a bit dodgy because I'm not shifting the functional arguments
    dAdr = first_r_derivative(A0, dr)

    dGc1_0 = dGdr_for_RK4_G1(rvals, nN0, U0, A0, dAdr, P0, epsilon, fa)
    dGc3_0 = dGdr_for_RK4_G3(rvals, nN0, U0, A0, dAdr, P0, epsilon, fa)
    dGc1_half = dGdr_for_RK4_G1(rvals + np.ones(np.shape(rvals))*dr/2, nN0, U0, A0, dAdr, P0, epsilon, fa)
    if np.isnan(dGc1_half).any():
        print("dGc1_half in solve_G_constraint_fast is nan")
        print(dGc1_half)
    dGc3_half = dGdr_for_RK4_G3(rvals + np.ones(np.shape(rvals))*dr/2, nN0, U0, A0, dAdr, P0, epsilon, fa)
    if np.isnan(dGc3_half).any():
        print("dGc3_half in solve_G_constraint_fast is nan")
        print(dGc3_half)
    dGc1_1 = dGdr_for_RK4_G1(rvals + np.ones(np.shape(rvals))*dr, nN0, U0, A0, dAdr, P0, epsilon, fa)
    dGc3_1 = dGdr_for_RK4_G3(rvals + np.ones(np.shape(rvals))*dr, nN0, U0, A0, dAdr, P0, epsilon, fa)
    
    # steps through until we reach the center
    for i in range(1, len(rvals)):
        G1[i] = solve_G_next_step_RK4_fast(rvals[i-1], dGc1_0[i-1], dGc3_0[i-1], dGc1_half[i-1], dGc3_half[i-1], dGc1_1[i-1], dGc3_1[i-1], G1[i-1], nN0[i-1], U0[i-1], A0[i-1], P0[i-1], epsilon, fa, dr)
        if np.isnan(G1[i]).any():
            print("broke at step : " + str(i))
    # sets the central F value so that F is flat across the origin
    Mtemp = rvals[-2] * (1.0 - G1[-2]**(-2.0)) / 2.0
    G1[-1] = np.sqrt(1.0 / (1.0 - 2.0 * Mtemp / rvals[-1]))
    #G1[1] = (8.0 + G1[2]) / 9.0
    
    return G1.astype(np.float64);
  
# now solve F constraint
def dFdr_for_RK4(r, G0, nN0, U0, A0, dAdr, P0, epsilon, fa):
    # equal to dFdr / F
    dFdronF = pow(r,-1)*(0.5 - 4.209734155810323e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 4.209734155810323e-38*pow(fa,2)*pow(P0,2)*pow(r,2) + (-0.5 + 4.209734155810323e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 4.209734155810323e-38*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,2) + pow(G0,2)*(-0.5 + 2.592333749140991e-41*epsilon*pow(r,2) - 8.419468311620646e-38*EOS(NRHO(nN0))*pow(r,2) - 1.8330567731163412e-41*epsilon*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + pow(U0,2)*(0.5 - 2.592333749140991e-41*epsilon*pow(r,2) - 8.419468311620646e-38*NRHO(nN0)*pow(r,2) + nN0*pow(r,2)*(4.967486303856181e-39 - 3.5125432509080046e-39*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 1.8330567731163412e-41*epsilon*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))))*pow(-1. + pow(U0,2),-1)
    return dFdronF.astype(np.float64)

def solve_F_next_step_RK4_fast(r, F0, dFdr_function_r0, dFdr_function_rhalf, dFdr_function_r1, G0, nN0, U0, A0, P0, epsilon, fa, dr):
    k1 = dFdr_function_r0 * F0
    k2 = dFdr_function_rhalf * (F0-k1*dr/2)
    k3 = dFdr_function_rhalf * (F0-k2*dr/2)
    k4 = dFdr_function_r1 * (F0-k3*dr)
    ans = F0 - (k1/6 + k2/3 + k3/3 + k4/6) * dr
    return ans

def solve_F_constraint_fast(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    F1 = np.zeros(np.shape(rvals))
    # sets initial condition at the very exterior
    F1[-1] = 1.0/G0[-1]
    # calculate the dFdr / F at different r values
    # this is maybe a bit dodgy because I'm not shifting the functional arguments
    dAdr = first_r_derivative(A0, dr)
    dFdr_function_r0 = np.zeros(np.shape(rvals))
    dFdr_function_rhalf = np.zeros(np.shape(rvals))
    dFdr_function_r1 = np.zeros(np.shape(rvals))
    dFdr_function_r0[1:] = dFdr_for_RK4(rvals[1:], G0[1:], nN0[1:], U0[1:], A0[1:], dAdr[1:], P0[1:], epsilon, fa)
    dFdr_function_rhalf[1:] = dFdr_for_RK4(rvals[1:] - np.ones(np.shape(rvals[1:]))*dr/2.0, G0[1:], nN0[1:], U0[1:], A0[1:], dAdr[1:], P0[1:], epsilon, fa)
    dFdr_function_r1[1:] = dFdr_for_RK4(rvals[1:] - np.ones(np.shape(rvals[1:]))*dr, G0[1:], nN0[1:], U0[1:], A0[1:], dAdr[1:], P0[1:], epsilon, fa)

    # steps through until we reach the center
    for i in range(2, len(rvals)):
        F1[-i] = solve_F_next_step_RK4_fast(rvals[-i+1], F1[-i+1], dFdr_function_r0[-i+1], dFdr_function_rhalf[-i+1], dFdr_function_r1[-i+1], G0[-i+1], nN0[-i+1], U0[-i+1], A0[-i+1], P0[-i+1], epsilon, fa, dr)
    # sets the central F value so that F is flat across the origin
    F1[0] = (9.0*F1[1] - F1[2]) / 8.0
    
    return F1.astype(np.float64);
