# SYSTEM IMPORTS
import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import time as time
import scipy.ndimage as scimage
import scipy.signal as sig
import math
from numpy import cos, sin, tan, roll, log
from scipy.interpolate import UnivariateSpline

# IMPORT MY OWN FUNCTIONS
from TOV_solver import *
from time_evolver import *
from constraint_solver import *
from data_saver import *



########################### ARGVS ####################################
RESOLUTION            = 20
tinit                 = 0.0
tfin                  = 200000.0
rinit                 = 5.0
rfin                  = 100005.0
Nt                    = 4*16000
Nr                    = 10000


############################ AXION POTENTIAL ########################################
def faxion(theta):
    return np.sqrt(1 - (4*(mu*md)/(mu+md)**2) * np.sin(theta/2)**2);

def V_tilde(theta, R0, eps):
    mN = MNfnc(R0)
    return (2*sigmaN*R0/mN) * (faxion(theta) - 1) + eps * mpi**2 * fpi**2 * (1 - faxion(theta)) * GeV2m**4 / hbarGeo**3



#################### TOV SOLVER ############################
# TOV initial state solver
def kGr_fnc(r, rho, G):
    return ((8*np.pi*r**2 * rho - 1)*G**3 + G) / (2*r);

def kRr_fnc(r, rho, G):
    return -(dEOSdrho(rho)**(-1)) * ((8*np.pi*r**2 * EOS(rho) + 1) * G**2 - 1) * (EOS(rho) + rho) / (2*r);

def solve_for_R_and_G_next_step(r, R0, G0, dr):
    # get k1
    k1Gr = kGr_fnc(r, R0, G0)
    k1Rr = kRr_fnc(r, R0, G0)
    # get k2
    k2Gr = kGr_fnc(r + dr/2, R0 + k1Rr*dr/2, G0 + k1Gr*dr/2)
    k2Rr = kRr_fnc(r + dr/2, R0 + k1Rr*dr/2, G0 + k1Gr*dr/2)
    # get k3
    k3Gr = kGr_fnc(r + dr/2, R0 + k2Rr*dr/2, G0 + k2Gr*dr/2)
    k3Rr = kRr_fnc(r + dr/2, R0 + k2Rr*dr/2, G0 + k2Gr*dr/2)
    # get k4
    k4Gr = kGr_fnc(r + dr, R0 + k3Rr*dr, G0 + k3Gr*dr)
    k4Rr = kRr_fnc(r + dr, R0 + k3Rr*dr, G0 + k3Gr*dr)
    # update with Runge-Kutta
    G1 = G0 + (1/6)*k1Gr*dr + (1/3)*k2Gr*dr + (1/3)*k3Gr*dr + (1/6)*k4Gr*dr
    R1 = R0 + (1/6)*k1Rr*dr + (1/3)*k2Rr*dr + (1/3)*k3Rr*dr + (1/6)*k4Rr*dr
    return np.array([R1, G1]).astype(np.float64);

def solve_for_R_and_G(Rinitial, RHOCUT, rvals):
    Ginitial = 1.0
    R1vals = np.zeros(np.shape(rvals))
    G1vals = np.zeros(np.shape(rvals))
    R1vals[0] = Rinitial
    G1vals[0] = Ginitial
    i = 1

    dr = rvals[1] - rvals[0]

    while R1vals[i-1] > RHOLS and i < Nr:
        newvals = solve_for_R_and_G_next_step(rvals[i-1], R1vals[i-1], G1vals[i-1], dr)
        R1vals[i] = newvals[0]
        G1vals[i] = newvals[1]
        i += 1
    
    i = i - 1
    while i < Nr:
        newvals = solve_for_R_and_G_next_step(rvals[i-1], R1vals[i-1], G1vals[i-1], dr)
        R1vals[i] = 0.0
        G1vals[i] = newvals[1]
        i += 1
        
    return np.concatenate((R1vals, G1vals)).astype(np.float64);

def kFr_fnc(r, rho, G, F):
    return F*((8*np.pi*r**2*EOS(rho) + 1)*G**2 - 1) / (2*r);

def solve_for_F_TOV_next_step(r, R0, G0, F0, dr):
    k1Fr = kFr_fnc(r, R0, G0, F0)
    k2Fr = kFr_fnc(r - dr/2, R0, G0, F0 - k1Fr*dr/2)
    k3Fr = kFr_fnc(r - dr/2, R0, G0, F0 - k2Fr*dr/2)
    k4Fr = kFr_fnc(r - dr, R0, G0, F0 - k3Fr*dr)
    return F0 - (k1Fr/6 + k2Fr/3 + k3Fr/3 + k4Fr/6) * dr;


def mass_TOV(r, G):
    return r*(1 - G**(-2)) / 2;

def solve_for_F_TOV(R0array, G0array, rvals):
    dr = rvals[1] - rvals[0]
    F1vals = np.zeros(np.shape(rvals))
    F1vals[-1] = np.sqrt(1 - 2*mass_TOV(rvals[-1], G0array[-1])/rvals[-1])
    for i in range(2, Nr):
        F1vals[-i] = solve_for_F_TOV_next_step(rvals[-i+1], R0array[-i+1], G0array[-i+1], F1vals[-i+1], dr)
    
    F1vals[0] = (9.0*F1vals[1] - F1vals[2]) / 8.0
    
    return F1vals;




####################################### TOV SOLVER WITH AXIONS ######################################
def kRr_fnc_TOV(r, R0, G0, A0):
    TOVpiece = -(dEOSdrho(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)))**(-1)) * ((8*np.pi*r**2 * EOS(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))) + 1) * G0**2 - 1) * (EOS(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))) + R0) / (2*r)
    return TOVpiece;

def kRr_fnc_nonTOV(r, R0, G0, A0vals, Ra0, pa0, dpadr, dAdr):
    nonTOVpiece = -(dEOSdrho(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0vals) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)))**(-1)) * ((8*np.pi*r**2 * pa0 + 1) * G0**2 - 1) * (pa0 + Ra0) / (2*r) - (dEOSdrho(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0vals) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)))**(-1)) * (2*r*dpadr + 4*(pa0-Ra0)) / (2*r) + (2*R0*sigmaN/MNfnc(R0)) * (((4*mu*md/(mu+md)**2)*sin(A0vals/2)*cos(A0vals/2))/faxion(A0vals)) * dAdr**2 / 2
    return nonTOVpiece;

def kRr_fnc_with_axion(r, R0, G0, A0vals, Ra0, pa0, dpadr, dAdr):
    return kRr_fnc_TOV(r, R0, G0, A0vals) + kRr_fnc_nonTOV(r, R0, G0, A0vals, Ra0, pa0, dpadr, dAdr);


def solve_for_R_next_step_with_axion(r, R0, G0, A0vals, Ra0, pa0, dpadr, dAdr):
    k1Rr = kRr_fnc_with_axion(r, R0, G0, A0vals, Ra0, pa0, dpadr, dAdr)
    k2Rr = kRr_fnc_with_axion(r + dr/2, R0 + k1Rr*dr/2, G0, A0vals, Ra0, pa0, dpadr, dAdr)
    k3Rr = kRr_fnc_with_axion(r + dr/2, R0 + k2Rr*dr/2, G0, A0vals, Ra0, pa0, dpadr, dAdr)
    k4Rr = kRr_fnc_with_axion(r + dr, R0 + k3Rr*dr, G0, A0vals, Ra0, pa0, dpadr, dAdr)
    # update with Runge-Kutta
    R1 = R0 + (1/6)*k1Rr*dr + (1/3)*k2Rr*dr + (1/3)*k3Rr*dr + (1/6)*k4Rr*dr
    return np.array([R1]).astype(np.float64);

def solve_for_R_with_axion(Rinitial, G0, A0vals, Ra0, pa0, RHOCUT):
    dpadr = first_r_derivative(pa0)
    dAdr = first_r_derivative(A0vals)
    R1vals = np.zeros(np.shape(rvals))
    R1vals[0] = Rinitial
    i = 1
    while R1vals[i-1] > RHOCUT and i < Nr:
        newvals = solve_for_R_next_step_with_axion(rvals[i-1], R1vals[i-1], G0[i-1], A0vals[i-1], Ra0[i-1], pa0[i-1], dpadr[i-1], dAdr[i-1])
        R1vals[i] = newvals[0]
        i += 1
    
    i = i - 1
    while i < Nr:
        R1vals[i] = 0.0
        i += 1
        
    return R1vals.astype(np.float64);



#################################### CONSTRAINT SOLVER ########################################
####################### Runge-Kutta for Spatial Integration #######################
def inside_density(r, G0, R0, A0, P0, dAdr, fa, epsilon):
    mN = MNfnc(R0)
    return R0 + pow(fa,2)*pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*pow(P0,2) - (pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*(-(pow(dAdr,2)*pow(fa,2)) + pow(fa,2)*pow(P0,2)))/2. - epsilon*pow(fpi,2)*pow(GeV2m,4)*pow(hbarGeo,-3)*pow(mpi,2)*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)) + R0*sigmaN*pow(mN,-1)*(-1 + (1 + epsilon)*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))

def axion_density(r, G0, R0, A0, P0, dAdr, fa, epsilon):
    mN = MNfnc(R0)
    return pow(fa,2)*pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*pow(P0,2) - (pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*(-(pow(dAdr,2)*pow(fa,2)) + pow(fa,2)*pow(P0,2)))/2. - epsilon*pow(fpi,2)*pow(GeV2m,4)*pow(hbarGeo,-3)*pow(mpi,2)*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)) + R0*sigmaN*pow(mN,-1)*(-1 + (1 + epsilon)*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))

def axion_radial_pressure(r, G0, R0, A0, P0, dAdr, fa, epsilon):
    mN = MNfnc(R0)
    return -pow(fa,2)*pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*pow(P0,2) + (pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*(-(pow(dAdr,2)*pow(fa,2)) + pow(fa,2)*pow(P0,2)))/2. + (- epsilon*pow(fpi,2)*pow(GeV2m,4)*pow(hbarGeo,-3)*pow(mpi,2)*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)) + R0*sigmaN*pow(mN,-1)*(-1 + (1 + epsilon)*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)));

def get_total_inside_mass_fast(rvals, G0array, R0array, A0array, P0array, dAdrarray, fa, epsilon):
    dr = rvals[1] - rvals[0]
    integral = (dr/2) * 4 * np.pi * (rvals**2 * inside_density(rvals, G0array, R0array, A0array, P0array, dAdrarray, fa, epsilon) + np.roll(rvals, -1, 0)**2 * inside_density(np.roll(rvals, -1, 0), np.roll(G0array, -1, 0), np.roll(R0array, -1, 0), np.roll(A0array, -1, 0), np.roll(P0array, -1, 0), np.roll(dAdrarray, -1, 0), fa, epsilon))
    integral = np.sum(integral) - integral[-1]
    return integral;

def k1Gr_G1(r, R0, A0, P0, dAdr):
    return 4.212591890594653e-38*r*pow(dAdr,2)*pow(fa,2) + 4.212591890594653e-38*r*pow(fa,2)*pow(P0,2) + pow(r,-1)/2.

def k1Gr_G3(r, R0, A0, P0, dAdr):
    return 6.662174619034264e-10*epsilon*r + 12.566370614359172*r*R0 - 0.5*pow(r,-1) - 6.662174619034264e-10*epsilon*r*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5) + r*R0*(-0.7891071341114869 + 0.7891071341114869*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5) + 0.789107134111487*epsilon*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))

def k1Gr_G5(r, R0, A0, P0, dAdr):
    return 0.0*rvals;

def k1Fr_F1(r, G0, R0, A0, P0, dAdr):
    return pow(r,-1)*(-0.5 + 4.212591890594653e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 4.212591890594653e-38*pow(fa,2)*pow(P0,2)*pow(r,2) + pow(G0,2)*(0.5 - 6.662174619034264e-10*epsilon*pow(r,2) + 12.566370614359172*EOS(R0*(0.9372048494885388 + 0.06279515051146117*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)))*pow(r,2) + 6.662174619034264e-10*epsilon*pow(r,2)*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5) + R0*pow(r,2)*(0.789107134111487 - 0.789107134111487*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5) - 0.789107134111487*epsilon*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))))

def solve_for_G_next_step_fast(r, G0, G1c1, G3c1, G5c1, G1c2, G3c2, G5c2, G1c3, G3c3, G5c3, G1c4, G3c4, G5c4):
    k1Gr = G1c1 * G0 + G3c1 * G0**3 + G5c1 * G0**5
    k2Gr = G1c2 * (G0 + k1Gr*dr/2) + G3c2 * (G0 + k1Gr*dr/2)**3 + G5c2 * (G0 + k1Gr*dr/2)**5
    k3Gr = G1c3 * (G0 + k2Gr*dr/2) + G3c3 * (G0 + k2Gr*dr/2)**3 + G5c3 * (G0 + k2Gr*dr/2)**5
    k4Gr = G1c4 * (G0 + k3Gr*dr) + G3c4 * (G0 + k3Gr*dr)**3 + G5c4 * (G0 + k3Gr*dr)**5
    return G0 + (k1Gr/6 + k2Gr/3 + k3Gr/3 + k4Gr/6) * dr;

def solve_for_F_next_step_fast(r, F0, F1c1, F1c2, F1c3, F1c4):
    k1Fr = F1c1 * F0
    k2Fr = F1c2 * (F0 - k1Fr*dr/2) 
    k3Fr = F1c3 * (F0 - k2Fr*dr/2)
    k4Fr = F1c4 * (F0 - k3Fr*dr)
    return F0 - (k1Fr/6 + k2Fr/3 + k3Fr/3 + k4Fr/6) * dr;

def solve_for_G_fast(G0point, R0array, A0array, P0array, dAdrarray, dRdrarray):
    R0array2 = (np.roll(R0array, -1, 0) + R0array)/2.0
    R0array2[-1] = R0array[-1]
    A0array2 = (np.roll(A0array, -1, 0) + A0array)/2.0
    A0array2[-1] = A0array[-1]
    P0array2 = (np.roll(P0array, -1, 0) + P0array)/2.0
    P0array2[-1] = P0array[-1]
    dAdrarray2 = (np.roll(dAdrarray, -1, 0) + dAdrarray)/2.0
    dAdrarray2[-1] = dAdrarray[-1]
    
    R0array4 = np.roll(R0array, -1, 0)
    R0array4[-1] = R0array[-1]
    A0array4 = np.roll(A0array, -1, 0)
    A0array4[-1] = A0array[-1]
    P0array4 = np.roll(P0array, -1, 0)
    P0array4[-1] = P0array[-1]
    dAdrarray4 = np.roll(dAdrarray, -1, 0)
    dAdrarray4[-1] = dAdrarray[-1]
    
    G1c1 = k1Gr_G1(rvals, R0array, A0array, P0array, dAdrarray)
    G3c1 = k1Gr_G3(rvals, R0array, A0array, P0array, dAdrarray)
    G5c1 = k1Gr_G5(rvals, R0array, A0array, P0array, dAdrarray)
    
    G1c2 = k1Gr_G1(rvals + np.ones(np.shape(rvals))*dr/2, R0array2, A0array2, P0array2, dAdrarray2)
    G3c2 = k1Gr_G3(rvals + np.ones(np.shape(rvals))*dr/2, R0array2, A0array2, P0array2, dAdrarray2)
    G5c2 = k1Gr_G5(rvals + np.ones(np.shape(rvals))*dr/2, R0array2, A0array2, P0array2, dAdrarray2)
    
    G1c4 = k1Gr_G1(rvals + np.ones(np.shape(rvals))*dr, R0array4, A0array4, P0array4, dAdrarray4)
    G3c4 = k1Gr_G3(rvals + np.ones(np.shape(rvals))*dr, R0array4, A0array4, P0array4, dAdrarray4)
    G5c4 = k1Gr_G5(rvals + np.ones(np.shape(rvals))*dr, R0array4, A0array4, P0array4, dAdrarray4)
    
    G1vals = np.zeros(np.shape(rvals))
    G1vals[0] = G0point
    for i in range(1, Nr):
        G1vals[i] = solve_for_G_next_step_fast(rvals[i-1], G1vals[i-1], G1c1[i-1], G3c1[i-1], G5c1[i-1], G1c2[i-1], G3c2[i-1], G5c2[i-1], G1c2[i-1], G3c2[i-1], G5c2[i-1], G1c4[i-1], G3c4[i-1], G5c4[i-1])
    
    G1vals[0] = (9.0*G1vals[1] - G1vals[2]) / 8.0
    
    return G1vals;

def solve_for_F_fast(G0array, R0array, A0array, P0array, dAdrarray):
    G0array2 = (np.roll(G0array, 1, 0) + G0array) / 2.0
    G0array2[0] = G0array[0]
    R0array2 = (np.roll(R0array, 1, 0) + R0array) / 2.0
    R0array2[0] = R0array[0]
    A0array2 = (np.roll(A0array, 1, 0) + A0array) / 2.0
    A0array2[0] = A0array[0]
    P0array2 = (np.roll(P0array, 1, 0) + P0array) / 2.0
    P0array2[0] = P0array[0]
    dAdrarray2 = (np.roll(dAdrarray, 1, 0) + dAdrarray) / 2.0
    dAdrarray2[0] = dAdrarray[0]
    
    G0array4 = np.roll(G0array, 1, 0)
    G0array4[0] = G0array[0]
    R0array4 = np.roll(R0array, 1, 0)
    R0array4[0] = R0array[0]
    A0array4 = np.roll(A0array, 1, 0)
    A0array4[0] = A0array[0]
    P0array4 = np.roll(P0array, 1, 0)
    P0array4[0] = P0array[0]
    dAdrarray4 = np.roll(dAdrarray, 1, 0)
    dAdrarray4[0] = dAdrarray[0]
    
    F1c1, F1c2, F1c4 = np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals))
    
    F1c1[1:] = k1Fr_F1(rvals[1:], G0array[1:], R0array[1:], A0array[1:], P0array[1:], dAdrarray[1:])
    
    F1c2[1:] = k1Fr_F1(rvals[1:] - np.ones(np.shape(rvals[1:]))*dr/2, G0array2[1:], R0array2[1:], A0array2[1:], P0array2[1:], dAdrarray2[1:])
    
    F1c4[1:] = k1Fr_F1(rvals[1:] - np.ones(np.shape(rvals[1:]))*dr, G0array4[1:], R0array4[1:], A0array4[1:], P0array4[1:], dAdrarray4[1:])
    
    F1vals = np.zeros(np.shape(rvals))
    integraltime = time.time()
    F1vals[-1] = np.sqrt(1 - 2*get_total_inside_mass_fast(rvals, G0array, R0array, A0array, P0array, dAdrarray) / rvals[-1])
    integraltotaltime = time.time() - integraltime
    #print(integraltotaltime)
    
    for i in range(2, Nr):
        F1vals[-i] = solve_for_F_next_step_fast(rvals[-i+1], F1vals[-i+1], F1c1[-i+1], F1c2[-i+1], F1c2[-i+1], F1c4[-i+1])
    
    F1vals[0] = (9.0*F1vals[1] - F1vals[2]) / 8.0
    
    return F1vals;

def metric_integrator(R0array, A0array, P0array):
    dAdrarray = first_r_derivative(A0array)
    dRdrarray = first_r_derivative(R0array)
    G1vals = solve_for_G_fast(1, R0array, A0array, P0array, dAdrarray, dRdrarray)
    F1vals = solve_for_F_fast(G1vals, R0array, A0array, P0array, dAdrarray)
    return np.concatenate((F1vals, G1vals)).astype(np.float64);







###################### TIME EVOLVER ##########################################################
###################### Runge-Kutta Integrator ########################
# the runge-kutta integrator will take initial conditions for the following six functions:
# U - radial 4 velocity scaled by G, U = G*ur
# R - baryon density
# A - axion field scaled by fa, A = a*fa
# P - axion velocity scaled by metric components, P = (G/F)*ak1
# it then steps them up using a 4th order Runge-Kutta
# for the spatial derivatives it uses a 4th order finite difference

def zero_below(myarrary, epsilon):
    result = myarrary.copy()
    result[result < epsilon] = 0.0
    return result;

def custom_sign(myarray):
    return np.where(myarray == 0, -1, np.sign(myarray));


# add in a changing baryon mass with a fixed baryon number density
# def density_with_axion_correction(NN0, A0):
#     return NN0*mN*GeV2m + NN0*sigmaN*GeV2m*(np.sqrt(1 - (4*mu*md / (mu+md)**2) * sin(A0/2)) - 1);

def sponge_function(r):
    return np.heaviside(r - rvals[Nr - NFRICTIONSITES], 1) * ((r - rvals[Nr - NFRICTIONSITES]) / (dr*FRICTIONSITESCALE))**3 / np.sqrt(1 + ((r - rvals[Nr - NFRICTIONSITES])/ (dr*FRICTIONSITESCALE))**6);

def kiG_function(r, F0, G0, R0, A0, P0):
    dAdr  = first_r_derivative(A0)
    k1G = 4*dAdr*F0*P0*Pi*r*pow(fa,2)*pow(GeV2m,2)*pow(hbarGeo,-1)
    return k1G.astype(np.float64);

def kiA_function(r, F0, G0, R0, A0, P0):
    k1Aarray = (F0*fa*P0*pow(G0,-1)) / fa
    #print(k1Aarray)
    return k1Aarray.astype(np.float64);

def kiP_function_nonzero(r, F0, G0, R0, A0, P0):
    mN = MNfnc(R0)
    dAdr  = first_r_derivative(A0)
    dAdr2 = second_r_derivative(A0)
    k1ParrayNonZero = (F0*pow(G0,-1)*pow(r,-1)*(dAdr*fa + dAdr2*fa*r + pow(fa,-1)*pow(G0,2)*pow(GeV2m,-2)*pow(hbarGeo,-3)*pow(mN,-1)*pow(2*md*mu*cos(A0) + pow(md,2) + pow(mu,2),-1)*(dAdr*mN*pow(fa,2)*pow(GeV2m,2)*(2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*(pow(hbarGeo,3) + 4*Pi*(-R0 + EOS(R0*(1 + sigmaN*pow(mN,-1)*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)))))*pow(hbarGeo,3)*pow(r,2) + 8*epsilon*Pi*pow(fpi,2)*pow(GeV2m,4)*pow(mpi,2)*pow(r,2)*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5))) - epsilon*hbarGeo*md*mN*mu*r*pow(fpi,2)*pow(GeV2m,4)*pow(mpi,2)*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)*sin(A0) + r*R0*sigmaN*pow(hbarGeo,3)*(-8*dAdr*Pi*r*pow(fa,2)*pow(GeV2m,2)*(2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*(-1 + pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5) + epsilon*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)) + (1 + epsilon)*hbarGeo*md*mu*pow((2*md*mu*cos(A0) + pow(md,2) + pow(mu,2))*pow(md + mu,-2),0.5)*sin(A0))))) / fa - sponge_function(r)*ZETAFRICTION*P0
    return k1ParrayNonZero.astype(np.float64);

def kiP_function_zero(r, F0, G0, R0, A0, P0, alimitcut):
    k1ParrayCoefficient = 0.0
    SinOverCos = (-2.0 * custom_sign(A0 - np.ones(np.shape(A0)) * np.pi))
    np.divide(sin(A0), myabs(cos(A0/2.0)), out=SinOverCos, where=(abs(A0 - np.ones(np.shape(A0)) * np.pi) >= alimitcut))
    k1ParrayZero = k1ParrayCoefficient * SinOverCos / fa
    
    return k1ParrayZero.astype(np.float64);

def kiP_function(r, F0, G0, R0, A0, P0):
    k1Parray = kiP_function_nonzero(r, F0, G0, R0, A0, P0) + kiP_function_zero(r, F0, G0, R0, A0, P0, ALIMITCUT)
    
    return k1Parray.astype(np.float64);

def matter_and_G_integrator(r, F0, G0, R0, A0, P0, indexflag):
    k1A = kiA_function(r, F0, G0, R0, A0, P0)
    k1P = kiP_function(r, F0, G0, R0, A0, P0)
    k1G = kiG_function(r, F0, G0, R0, A0, P0)
    #k1G[0] = 0.0
    #k1G[1] = 0.0
    
    # get k2 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayhalf1 = first_r_derivative(A0 + k1A*dt/2)
        Fhalf1 = solve_for_F_fast(G0 + k1G*dt/2, R0, A0 + k1A*dt/2, P0 + k1P*dt/2, dAdrarrayhalf1)
        #Fhalf1[1] = Fhalf1[0]
    else:
        Fhalf1 = F0
        #Fhalf1[1] = Fhalf1[0]
        
    k2G = kiG_function(r, Fhalf1, G0 + k1G*dt/2, R0, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2A = kiA_function(r, Fhalf1, G0 + k1G*dt/2, R0, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2P = kiP_function(r, Fhalf1, G0 + k1G*dt/2, R0, A0 + k1A*dt/2, P0 + k1P*dt/2)
    #k2G[0] = 0.0
    #k2G[1] = 0.0
    
    # get k3 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayhalf2 = first_r_derivative(A0 + k2A*dt/2)
        Fhalf2 = solve_for_F_fast(G0 + k2G*dt/2, R0, A0 + k2A*dt/2, P0 + k2P*dt/2, dAdrarrayhalf2)
        #Fhalf2[1] = Fhalf2[0]
    else:
        Fhalf2 = F0
        #Fhalf2[1] = Fhalf2[0]
        
    k3G = kiG_function(r, Fhalf2, G0 + k2G*dt/2, R0, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3A = kiA_function(r, Fhalf2, G0 + k2G*dt/2, R0, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3P = kiP_function(r, Fhalf2, G0 + k2G*dt/2, R0, A0 + k2A*dt/2, P0 + k2P*dt/2)
    #k3G[0] = 0.0
    #k3G[1] = 0.0
    
    # get k4 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayfull = first_r_derivative(A0 + k3A*dt)
        Ffull = solve_for_F_fast(G0 + k3G*dt, R0, A0 + k3A*dt, P0 + k3P*dt, dAdrarrayfull)
        #Ffull[1] = Ffull[0]
    else:
        Ffull = F0
        #Ffull[1] = Ffull[0]
        
    k4G = kiG_function(r, Ffull, G0 + k3G*dt, R0, A0 + k3A*dt, P0 + k3P*dt)
    k4A = kiA_function(r, Ffull, G0 + k3G*dt, R0, A0 + k3A*dt, P0 + k3P*dt)
    k4P = kiP_function(r, Ffull, G0 + k3G*dt, R0, A0 + k3A*dt, P0 + k3P*dt)
    #k4G[0] = 0.0
    #k4G[1] = 0.0
    
    # update with Runge-Kutta
    G1 = G0 + (1/6)*k1G*dt + (1/3)*k2G*dt + (1/3)*k3G*dt + (1/6)*k4G*dt
    A1 = A0 + (1/6)*k1A*dt + (1/3)*k2A*dt + (1/3)*k3A*dt + (1/6)*k4A*dt
    P1 = P0 + (1/6)*k1P*dt + (1/3)*k2P*dt + (1/3)*k3P*dt + (1/6)*k4P*dt
    
    G1[0] = (9.0*G1[1] - G1[2]) / 8.0
    A1[0] = (9.0*A1[1] - A1[2]) / 8.0
    P1[0] = (9.0*P1[1] - P1[2]) / 8.0
    
    #G1[-1] = (G1[-2] - (dr/dt)*G0[-1]) / (1 - (dr/dt))
    #A1[-1] = (A1[-2] - (dr/dt)*A0[-1]) / (1 - (dr/dt))
    #P1[-1] = (P1[-2] - (dr/dt)*P0[-1]) / (1 - (dr/dt))
    
    return np.concatenate((A1, P1, G1)).astype(np.float64);


# define the constants you need for RKF45
A1 = 0.
A2 = 2/9
A3 = 1/3
A4 = 3/4
A5 = 1.
A6 = 5.6

B21 = 2/9
B31 = 1/12
B32 = 1/4
B41 = 69/128
B42 = -243/128
B43 = 135/64
B51 = -17/12
B52 = 27/4
B53 = -27/5
B54 = 16/15
B61 = 65/432
B62 = -5/16
B63 = 13/16
B64 = 4/27
B65 = 5/144

C1 = 1/9
C2 = 0.
C3 = 9/20
C4 = 16/45
C5 = 1/12

CH1 = 47/450
CH2 = 0.
CH3 = 12/25
CH4 = 32/225
CH5 = 1/30
CH6 = 6/25

CT1 = 1/150
CT2 = 0.
CT3 = -3/100
CT4 = 16/75
CT5 = 1/20
CT6 = -6/25

truncation_error_strings = ["G", "A", "P"]


# now define the RHF45 integrator
def matter_and_G_integrator_RKF45(r, F0, G0, R0, A0, P0, indexflag, tstep, acceptable_errors_fraction, verbose):
    acceptable_errors = acceptable_errors_fraction * np.array([1.0, np.pi, np.pi/100])
    
    k1A = tstep*kiA_function(r, F0, G0, R0, A0, P0)
    k1P = tstep*kiP_function(r, F0, G0, R0, A0, P0)
    k1G = tstep*kiG_function(r, F0, G0, R0, A0, P0)

    # solve for the quarter step of the F2 metric function
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray2 = first_r_derivative(A0 + B21*k1A)
        Fint2 = solve_for_F_fast(G0 + B21*k1G, R0, A0 + B21*k1A, P0 + B21*k1P, dAdrarray2)
    else:
        Fint2 = F0
        
        
    k2G = tstep*kiG_function(r, Fint2, G0 + B21*k1G, R0, A0 + B21*k1A, P0 + B21*k1P)
    k2A = tstep*kiA_function(r, Fint2, G0 + B21*k1G, R0, A0 + B21*k1A, P0 + B21*k1P)
    k2P = tstep*kiP_function(r, Fint2, G0 + B21*k1G, R0, A0 + B21*k1A, P0 + B21*k1P)

    # solve for F3 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray3 = first_r_derivative(A0 + B31*k1A + B32*k2A)
        Fint3 = solve_for_F_fast(G0 + B31*k1G + B32*k2G, R0, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P, dAdrarray3)
    else:
        Fint3 = F0
        
    
    k3G = tstep*kiG_function(r, Fint3, G0 + B31*k1G + B32*k2G, R0, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    k3A = tstep*kiA_function(r, Fint3, G0 + B31*k1G + B32*k2G, R0, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    k3P = tstep*kiP_function(r, Fint3, G0 + B31*k1G + B32*k2G, R0, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)

    # solve for F4 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray4 = first_r_derivative(A0 + B41*k1A + B42*k2A + B43*k3A)
        Fint4 = solve_for_F_fast(G0 + B41*k1G + B42*k2G + B43*k3G, R0, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P, dAdrarray4)
    else:
        Fint4 = F0
        
    
    k4G = tstep*kiG_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, R0, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    k4A = tstep*kiA_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, R0, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    k4P = tstep*kiP_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, R0, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    
    # solve for F5 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray5 = first_r_derivative(A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A)
        Fint5 = solve_for_F_fast(G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, R0, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P, dAdrarray5)
    else:
        Fint5 = F0
        
        
    k5G = tstep*kiG_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, R0, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    k5A = tstep*kiA_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, R0, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    k5P = tstep*kiP_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, R0, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)

    # solve for F6
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray6 = first_r_derivative(A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A)
        Fint6 = solve_for_F_fast(G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, R0, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P, dAdrarray6)
    else:
        Fint6 = F0
        
        
    k6G = tstep*kiG_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, R0, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    k6A = tstep*kiA_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, R0, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    k6P = tstep*kiP_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, R0, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)

    
    # update with Runge-Kutta
    G1 = G0 + CH1*k1G + CH2*k2G + CH3*k3G + CH4*k4G + CH5*k5G + CH6*k6G
    A1 = A0 + CH1*k1A + CH2*k2A + CH3*k3A + CH4*k4A + CH5*k5A + CH6*k6A
    P1 = P0 + CH1*k1P + CH2*k2P + CH3*k3P + CH4*k4P + CH5*k5P + CH6*k6P
    
    truncation_error_G = max(abs(CT1*k1G + CT2*k2G + CT3*k3G + CT4*k4G + CT5*k5G + CT6*k6G))
    truncation_error_A = max(abs(CT1*k1A + CT2*k2A + CT3*k3A + CT4*k4A + CT5*k5A + CT6*k6A))
    truncation_error_P = max(abs(CT1*k1P + CT2*k2P + CT3*k3P + CT4*k4P + CT5*k5P + CT6*k6P))
    
    truncation_errors = np.array([truncation_error_G, truncation_error_A, truncation_error_P])
    print(truncation_errors)
    
    hnewG = 0.9 * tstep * (acceptable_errors[0]/max(truncation_error_G, acceptable_errors[0]/10000000))**(1/5)
    hnewA = 0.9 * tstep * (acceptable_errors[1]/max(truncation_error_A, acceptable_errors[1]/10000000))**(1/5)
    hnewP = 0.9 * tstep * (acceptable_errors[2]/max(truncation_error_P, acceptable_errors[2]/10000000))**(1/5)
    
    hnews = [hnewG, hnewA, hnewP]
    hnew = min(hnews)
    hnewindex = hnews.index(hnew)
    
    if verbose == 1:
        #print("\n")
        print("TE problem variable: " + str(truncation_error_strings[hnewindex]))
        #print("\n")
    
    if truncation_errors[hnewindex] > acceptable_errors[hnewindex]:
        return matter_and_G_integrator_RKF45(r, F0, G0, R0, A0, P0, indexflag, hnew, acceptable_errors_fraction, verbose);
    else:
        return [np.concatenate((A1, P1, G1)).astype(np.float64), hnew];





############################################## DATA SAVERS ##############################################
def save_total_data(Fdata, Gdata, Adata, Pdata, times, radii, preamblevars, directory):
    # preamblevars will contain the following variables
    # [Rcentral in m^-2, fa in GeV, ma in GeV, epsilon, INTERMEDIATEMETRICINTEGRATOR]
    np.savetxt(directory + "/F-solution" + ".txt", Fdata, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    print("saved F")
    np.savetxt(directory + "/G-solution" + ".txt", Gdata, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    print("saved G")
    np.savetxt(directory + "/A-solution" + ".txt", Adata, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    print("saved A")
    np.savetxt(directory + "/P-solution" + ".txt", Pdata, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    print("saved P")
    np.savetxt(directory + "/times.txt", times, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    np.savetxt(directory + "/radii.txt", radii, fmt='%.18e', delimiter=',', newline='\n', header=('Rc ' + str(preamblevars[0]) + ' fa ' + str(preamblevars[1]) + ' ma ' + str(preamblevars[2]) + ' epsilon ' + str(preamblevars[3]) + ' INTERMEDIATEMATTERINTEGRATOR ' + str(preamblevars[4])))
    print("saved times and radii")

    return;


plot_options = 1

colors = [(101/255, 144/255, 255/255), (121/255, 94/255, 240/255), (221/255, 38/255, 129/255), (255/255, 97/255, 2/255), (254/255, 176/255, 1/255), (200/255, 200/255, 200/255), (13/255, 35/255, 65/255), (0/255, 130/255, 61/255)]

def make_plot_slide(totaldensity, barebaryondensity, fs, fs0, gs, gs0, ayes, ayes0, pis, pis0, plots, index, showflag, saveflag, directory):
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 4))

    ax[0].plot(rvals / 10**3, fs**2, color=colors[0], label='$g_{tt}(t, r)$')
    ax[0].plot(rvals / 10**3, fs0**2, '--', color=colors[1], label='$g_{tt}(t = 0, r)$')
    ax[0].plot(rvals / 10**3, -gs**2, color=colors[2], label='$g_{rr}(t, r)$')
    ax[0].plot(rvals / 10**3, -gs0**2, '--', color=colors[3], label='$g_{rr}(t = 0, r)$')
    ax[0].plot(rvals / 10**3, np.log(fs0/gs0), '--', color=colors[5], label='$\\log{\\sqrt{\\frac{g_{tt}}{g_{rr}}}}$ at $t = 0$')
    ax[0].plot(rvals / 10**3, np.log(fs/gs), '-', color=colors[4], label='$\\log{\\sqrt{\\frac{g_{tt}}{g_{rr}}}}$')
    ax[0].set_xlim(rvals[0]/ 10**3, rvals[-int(Nr/2)]/ 10**3)
    ax[0].set_ylim(-2.8, 1.3)
    ax[0].set_xlabel(r'$r$ (km)', fontname="serif", fontsize=12)
    ax[0].set_ylabel(r'Metric Values [a.u.]', fontname="serif", fontsize=12)
    ax[0].legend(prop={'family': 'serif'})

    ax[1].plot(rvals / 10**3, ayes, color=colors[0])
    #ax[1].plot(rvals / 10**3, ayes0, '--', color=colors[1])
    #ax[1].plot(rvals / 10**3, -ayes0 + np.ones(np.shape(ayes0))*2.*np.pi, '--', color=colors[1])
    ax[1].plot(rvals / 10**3, np.ones(np.shape(rvals))*np.pi, '--', color=colors[2])
    ax[1].set_xlim(rvals[0]/ 10**3, rvals[-int(Nr/2)]/ 10**3)
    ax[1].set_ylim(-0.5, 6.5)
    ax[1].set_xlabel(r'$r$ (km)', fontname="serif", fontsize=12)
    ax[1].set_ylabel(r'Axion Field $a(t, r) f_a$ [a.u.]', fontname="serif", fontsize=12)

    ax[2].plot(rvals / 10**3, totaldensity, color=colors[0])
    ax[2].plot(rvals / 10**3, barebaryondensity, '--', color=colors[1])
    #ax[2].plot(rvals / 10**3, ayes0, '--', color=colors[1])
    #ax[2].plot(rvals / 10**3, -ayes0 + np.ones(np.shape(ayes0))*2.*np.pi, '--', color=colors[1])
    #ax[2].plot(rvals / 10**3, np.ones(np.shape(rvals))*np.pi, '--', color=colors[2])
    ax[2].set_xlim(rvals[0]/ 10**3, rvals[-int(Nr/2)]/ 10**3)
    ax[2].set_ylim(-1e-11, 3e-9)
    ax[2].set_xlabel(r'$r$ (km)', fontname="serif", fontsize=12)
    ax[2].set_ylabel(r'Axion Energy $\rho_a(t, r) f_a$ [a.u.]', fontname="serif", fontsize=12)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.6)

    plt.tight_layout()
    if saveflag == 1:
        plt.savefig(directory + "/solution-" + str(str(int(index)).zfill(4)) + ".png", dpi=150, facecolor='white', transparent=False)
    if showflag == 1:
        plt.show()
    plt.close()



############################################### extra functions for analysis ############################################
def get_frequencies_1D(data, times, index, temporl_cut_index):
    dataFFT = np.fft.rfft(data[temporl_cut_index:, index])
    Nt = int(len(times) / RESOLUTION)
    dt = times[RESOLUTION] - times[0]

    angular_frequencies = 2*np.pi * np.fft.rfftfreq(Nt - temporl_cut_index, dt / clight) # gives frequencies in seconds

    return [angular_frequencies, np.abs(dataFFT)];


def get_fft_peaks_and_widths(fs, As):
    peak_indices = sig.find_peaks(As, distance=20, width=0.1, prominence=0.1)
    #print(peak_indices[1])
    df = fs[1] - fs[0]
    peak_frequencies = []
    peak_amplitudes  = []
    peak_widths = []
    peak_width_heights = []
    peak_width_fractional_heights = []
    fwhms = []
    for i in range(len(peak_indices[0])):
        peak_frequencies.append(fs[peak_indices[0][i]])
        peak_amplitudes.append(As[peak_indices[0][i]])
        peak_widths.append(peak_indices[1]["widths"][i]*df)
        peak_width_heights.append(peak_indices[1]["width_heights"][i])
        peak_width_fractional_heights.append(peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]])
        fwhms.append((peak_indices[1]["widths"][i]*df)*(peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]]) / (1 - (peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]])))
        
    
    return [peak_frequencies, peak_amplitudes, fwhms];



def find_radius(R0vals, rvals):
    rindex = 0
    while R0vals[rindex] > RHOLS:
        rindex += 1
    
    return rvals[rindex];


###################################### functions for calculating tidal deformability ##############################################
def zeta_perturbation(Fvals, F0vals):
    return 1 - (Fvals / F0vals)**2;

def delta_perturbation(Gvals, G0vals):
    return 1 - (Gvals / G0vals)**2;

def find_radius_with_index(R0vals, rvals):
    rindex = 0
    while R0vals[rindex] > RHOLS:
        rindex += 1
    
    return [rvals[rindex], rindex];

# tidal deformability calculator
def ky0_fnc(r, Y00, Z0, D0, M0, R0, dRdr0):
    ky0 = pow(r,-1)*pow(-2*M0 + r,-2)*(2*r*pow(M0,2)*(-6 + 20*Pi*R0*pow(r,2) + Y00*(1 + 2*Y00 - 4*Pi*R0*pow(r,2)) + 8*dRdr0*Pi*pow(r,3) + 4*Pi*EOS(R0)*pow(r,2)*(15 + Y00 - 2*pow(Y00,2))) - 4*pow(M0,3)*(-1 + pow(Y00,2)) + M0*pow(r,2)*(6 - 20*Pi*R0*pow(r,2) - Y00*(1 + Y00 - 4*Pi*R0*pow(r,2)) - 16*dRdr0*Pi*pow(r,3) + 4*Pi*EOS(R0)*pow(r,2)*(-21 + Y00 - 8*Pi*R0*(-5 + Y00)*pow(r,2) + 4*pow(Y00,2)) + 32*(15 + Y00)*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2)) + 4*Pi*pow(r,5)*(dRdr0*r + EOS(R0)*(6 - Y00 + 4*Pi*R0*(-5 + Y00)*pow(r,2) - 4*Pi*(9 + Y00)*EOS(R0)*pow(r,2) - pow(Y00,2) + 64*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2))))*pow(M0 + 4*Pi*EOS(R0)*pow(r,3),-1)
    return ky0;

def ky1_fnc(r, Y10, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0):
    ky1 = pow(r,-1)*pow(-2*M0 + r,-2)*(-2*(-D0 + Y00*(D0 + 4*Y10 - Z0) + Z0)*pow(M0,4) + r*pow(M0,3)*(-10*D0 - D0*Y00 - 2*Z0 - Y00*Z0 + 24*D0*Pi*R0*pow(r,2) + 104*Pi*Ra0*pow(r,2) + 8*Pa0*Pi*(-3 + Y00)*pow(r,2) - 8*D0*Pi*R0*Y00*pow(r,2) - 40*Pi*Ra0*Y00*pow(r,2) + 16*Pi*R0*Z0*pow(r,2) + 8*D0*Pi*EOS(R0)*pow(r,2) - 48*D0*Pi*Y00*EOS(R0)*pow(r,2) + 32*Pi*Z0*EOS(R0)*pow(r,2) + 24*Pi*Y00*Z0*EOS(R0)*pow(r,2) + Y10*(2 + 8*Y00 - 8*Pi*R0*pow(r,2) + 8*Pi*(1 - 8*Y00)*EOS(R0)*pow(r,2)) + 16*dRadr0*Pi*pow(r,3) + 16*D0*dRdr0*Pi*pow(r,3)) - pow(M0,2)*pow(r,2)*(-4*D0 - D0*Y00 - 2*Z0 + 4*D0*Pi*R0*pow(r,2) + 60*Pi*Ra0*pow(r,2) - 4*D0*Pi*R0*Y00*pow(r,2) - 36*Pi*Ra0*Y00*pow(r,2) + 8*Pi*R0*Z0*pow(r,2) + 76*D0*Pi*EOS(R0)*pow(r,2) - 8*D0*Pi*Y00*EOS(R0)*pow(r,2) + 32*Pi*Z0*EOS(R0)*pow(r,2) + 12*Pi*Y00*Z0*EOS(R0)*pow(r,2) + 16*dRadr0*Pi*pow(r,3) + 24*D0*dRdr0*Pi*pow(r,3) + 4*Pa0*Pi*pow(r,2)*(-9 + Y00*(5 + 16*Pi*EOS(R0)*pow(r,2)) + 16*dRdr0*Pi*pow(r,3)) - 256*D0*R0*EOS(R0)*pow(Pi,2)*pow(r,4) - 768*Ra0*EOS(R0)*pow(Pi,2)*pow(r,4) + 64*D0*R0*Y00*EOS(R0)*pow(Pi,2)*pow(r,4) + 192*Ra0*Y00*EOS(R0)*pow(Pi,2)*pow(r,4) - 128*R0*Z0*EOS(R0)*pow(Pi,2)*pow(r,4) - 64*dRadr0*EOS(R0)*pow(Pi,2)*pow(r,5) + 288*D0*Y00*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2) - 384*Z0*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2) - 96*Y00*Z0*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2) + Y10*(1 + 2*Y00 - 4*Pi*R0*pow(r,2) + 4*Pi*EOS(R0)*pow(r,2)*(-3 - 16*Y00 + 16*Pi*R0*pow(r,2)) + 64*(-1 + 2*Y00)*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2))) - 4*M0*Pi*pow(r,5)*(-(dRadr0*r) - 3*D0*dRdr0*r - 2*Ra0 + 2*Ra0*Y00 + 2*EOS(R0)*(-2*Z0 + 44*Pi*Ra0*pow(r,2) - 20*Pi*Ra0*Y00*pow(r,2) + 8*Pi*R0*Z0*pow(r,2) + Y10*(1 + 2*Y00 - 4*Pi*R0*pow(r,2)) - D0*(4 + Y00 - 4*Pi*R0*pow(r,2) + 4*Pi*R0*Y00*pow(r,2)) + 8*dRadr0*Pi*pow(r,3)) + 4*Pi*pow(r,2)*(2*(4*Pi*(-13*Ra0 + R0*Y10)*pow(r,2) + D0*(4 - 28*Pi*R0*pow(r,2)) + Z0*(5 - 8*Pi*R0*pow(r,2))) + Y00*(-8*Y10 + 3*Z0 + 8*Pi*Ra0*pow(r,2) + D0*(-7 + 8*Pi*R0*pow(r,2))))*pow(EOS(R0),2) + 2*Pa0*(1 - Y00 + 4*Pi*(-5 + Y00)*EOS(R0)*pow(r,2) - 8*dRdr0*Pi*pow(r,3) + 16*(-5 + 3*Y00)*pow(Pi,2)*pow(r,4)*pow(EOS(R0),2)) - 32*(D0*(1 - 4*Y00) + Y10 + (8 + Y00)*Z0)*pow(Pi,2)*pow(r,4)*pow(EOS(R0),3)) + 2*Pi*pow(r,7)*(8*Pi*r*(dRadr0*r + 2*Pa0*(-1 + Y00) - 2*Ra0*(-1 + Y00))*EOS(R0) - dRdr0*(D0 + 8*Pa0*Pi*pow(r,2)) + 8*Pi*r*(4*D0 + D0*Y00 + 2*Z0 - 4*D0*Pi*R0*pow(r,2) - 28*Pi*Ra0*pow(r,2) + 4*D0*Pi*R0*Y00*pow(r,2) + 4*Pi*Ra0*Y00*pow(r,2) + 4*Pa0*Pi*(1 + 3*Y00)*pow(r,2) - 8*Pi*R0*Z0*pow(r,2) + Y10*(-1 - 2*Y00 + 4*Pi*R0*pow(r,2)))*pow(EOS(R0),2) + 32*pow(Pi,2)*(D0 + 4*D0*Y00 - Y10 - 4*Z0 - Y00*Z0 + 16*Pa0*Pi*pow(r,2) + 16*D0*Pi*R0*pow(r,2) + 16*Pi*Ra0*pow(r,2))*pow(r,3)*pow(EOS(R0),3) + 256*(D0 + Z0)*pow(Pi,3)*pow(r,5)*pow(EOS(R0),4)))*pow(M0 + 4*Pi*EOS(R0)*pow(r,3),-2)
    return ky1;

def solve_for_y0_next_step(r, Y00, Z0, D0, M0, R0, dRdr0, dr):
    # get k1
    k1yr = ky0_fnc(r, Y00, Z0, D0, M0, R0, dRdr0)
    # get k2
    k2yr = ky0_fnc(r + dr/2, Y00 + k1yr*dr/2, Z0, D0, M0, R0, dRdr0)
    # get k3
    k3yr = ky0_fnc(r + dr/2, Y00 + k2yr*dr/2, Z0, D0, M0, R0, dRdr0)
    # get k4
    k4yr = ky0_fnc(r + dr, Y00 + k3yr*dr, Z0, D0, M0, R0, dRdr0)
    # update with Runge-Kutta
    Y01 = Y00 + (1/6)*k1yr*dr + (1/3)*k2yr*dr + (1/3)*k3yr*dr + (1/6)*k4yr*dr
    return np.array(Y01).astype(np.float64);

def solve_for_y1_next_step(r, Y10, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0, dr):
    # get k1
    k1yr = ky1_fnc(r, Y10, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0)
    # get k2
    k2yr = ky1_fnc(r + dr/2, Y10 + k1yr*dr/2, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0)
    # get k3
    k3yr = ky1_fnc(r + dr/2, Y10 + k2yr*dr/2, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0)
    # get k4
    k4yr = ky1_fnc(r + dr, Y10 + k3yr*dr, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0)
    # update with Runge-Kutta
    Y11 = Y10 + (1/6)*k1yr*dr + (1/3)*k2yr*dr + (1/3)*k3yr*dr + (1/6)*k4yr*dr
    return np.array(Y11).astype(np.float64);

def solve_for_y0(rvals, Z0, D0, M0, R0, dRdr0):
    dr = rvals[1] - rvals[0]
    y0initial = 2
    y0vals = np.zeros(np.shape(rvals))
    Nr = np.shape(rvals)[0]
    y0vals[0] = y0initial
    i = 1
    while R0[i-1] > 0.0 and i < Nr:
        y0vals[i] = solve_for_y0_next_step(rvals[i-1], y0vals[i-1], Z0[i-1], D0[i-1], M0[i-1], R0[i-1], dRdr0[i-1], dr)
        i += 1
    
    return y0vals;

def solve_for_y1(rvals, Y00, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0):
    dr = rvals[1] - rvals[0]
    y1initial = (2*(3*D0[0] + Z0[0])*(3*EOS(R0[0]) + R0[0]) + 6*Pa0[0] - 6*Ra0[0]) / (5 * (3*EOS(R0[0]) + R0[0]))
    y1vals = np.zeros(np.shape(rvals))
    Nr = np.shape(rvals)[0]
    y1vals[0] = y1initial
    i = 1
    while R0[i-1] > 0.0 and i < Nr:
        y1vals[i] = solve_for_y1_next_step(rvals[i-1], y1vals[i-1], Y00[i-1], Z0[i-1], D0[i-1], M0[i-1], R0[i-1], Ra0[i-1], Pa0[i-1], dRdr0[i-1], dRadr0[i-1], dr)
        i += 1
    
    return y1vals;

def Lambda_baryon(Rstar, y0star, Mbaryon):
    return (16*(-(Rstar*(-2 + y0star)) + 2*Mbaryon*(-1 + y0star))*pow(Rstar,2)*pow(-2*Mbaryon + Rstar,2)*pow(2*Mbaryon*(2*Rstar*(-2 + 3*y0star)*pow(Mbaryon,3) + 4*(1 + y0star)*pow(Mbaryon,4) + 2*(13 - 11*y0star)*pow(Mbaryon,2)*pow(Rstar,2) + 3*Mbaryon*(-8 + 5*y0star)*pow(Rstar,3) - 3*(-2 + y0star)*pow(Rstar,4)) + 3*(Rstar*(-2 + y0star) - 2*Mbaryon*(-1 + y0star))*log(Rstar*pow(-2*Mbaryon + Rstar,-1))*pow(Rstar,2)*pow(-2*Mbaryon + Rstar,2),-1))/15.0;

def Lambda_axion(Rstar, y0star, Mbaryon, y1star, Maxion):
    return (8*(2*Mbaryon - Rstar)*pow(Mbaryon,-1)*pow(Rstar,2)*(Maxion*(2*Mbaryon - Rstar)*(-(Rstar*(-2 + y0star)) + 2*Mbaryon*(-1 + y0star))*log(1 - 2*Mbaryon*pow(Rstar,-1))*(2*(4*Rstar*(-2 + 3*y0star)*pow(Mbaryon,4) + 8*(1 + y0star)*pow(Mbaryon,5) + 4*(43 - 41*y0star)*pow(Mbaryon,3)*pow(Rstar,2) + 6*(-48 + 35*y0star)*pow(Mbaryon,2)*pow(Rstar,3) + 6*Mbaryon*(27 - 16*y0star)*pow(Rstar,4) + 15*(-2 + y0star)*pow(Rstar,5)) - 3*(Rstar*(-2 + y0star) - 2*Mbaryon*(-1 + y0star))*log(1 - 2*Mbaryon*pow(Rstar,-1))*pow(Rstar,2)*pow(-2*Mbaryon + Rstar,2)) + 4*Mbaryon*(4*Rstar*(Maxion*(-61 + (56 - 19*y0star)*y0star) + 2*Rstar*y1star)*pow(Mbaryon,5) + 8*(Maxion*(9 + (2 - 11*y0star)*y0star) - 4*Rstar*y1star)*pow(Mbaryon,6) + 32*y1star*pow(Mbaryon,7) + 2*Maxion*(404 + 5*y0star*(-142 + 63*y0star))*pow(Mbaryon,4)*pow(Rstar,2) + Maxion*(-1452 + (2252 - 841*y0star)*y0star)*pow(Mbaryon,3)*pow(Rstar,3) + 2*Maxion*(586 + y0star*(-772 + 247*y0star))*pow(Mbaryon,2)*pow(Rstar,4) - 6*Maxion*Mbaryon*(-2 + y0star)*(-36 + 23*y0star)*pow(Rstar,5) + 15*Maxion*pow(Rstar,6)*pow(-2 + y0star,2)))*pow(2*Mbaryon*(2*Rstar*(-2 + 3*y0star)*pow(Mbaryon,3) + 4*(1 + y0star)*pow(Mbaryon,4) + 2*(13 - 11*y0star)*pow(Mbaryon,2)*pow(Rstar,2) + 3*Mbaryon*(-8 + 5*y0star)*pow(Rstar,3) - 3*(-2 + y0star)*pow(Rstar,4)) + 3*(Rstar*(-2 + y0star) - 2*Mbaryon*(-1 + y0star))*log(Rstar*pow(-2*Mbaryon + Rstar,-1))*pow(Rstar,2)*pow(-2*Mbaryon + Rstar,2),-2))/15.;


def solve_tidal_deformabilities(rvals, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0, Mbaryon, Maxion):
    # first get the index at which the radius is located and get the radius
    radius_and_index = find_radius_with_index(R0, rvals)
    Rstar = radius_and_index[0]
    rindex = radius_and_index[1]

    # solve y0 and y1 ODEs and assign values to y0and y1 at Rstar
    y0vals = solve_for_y0(rvals, Z0, D0, M0, R0, dRdr0)
    y1vals = solve_for_y1(rvals, y0vals, Z0, D0, M0, R0, Ra0, Pa0, dRdr0, dRadr0)
    y0star = y0vals[rindex]
    y1star = y1vals[rindex]
    
    # solve for each contribution to tidal deformability
    LambdaB = Lambda_baryon(Rstar, y0star, Mbaryon)
    LambdaA = Lambda_axion(Rstar, y0star, Mbaryon, y1star, Maxion)
    
    return [LambdaB, LambdaA];





################################### define function for total analysis #########################################

def total_analysis_function(directory, index):
    # load in all the data that we have
    parameterfile = np.loadtxt("/u/wentzel4/axions-in-NS/SLy4-simulation/inputs/simulation_parameter_sets.csv", dtype='float', comments="#", delimiter=",")

    Rcentral = parameterfile[index, 0]
    epsilon  = parameterfile[index, 2]
    fa       = parameterfile[index, 1]
    ma       = np.sqrt(mu*md / (mu+md)**2) * mpi*fpi / fa

    Ftotal = np.load(directory + "/F-solution.npy", allow_pickle=False)
    Gtotal = np.load(directory + "/G-solution.npy", allow_pickle=False)
    Atotal = np.load(directory + "/A-solution.npy", allow_pickle=False, max_header_size=1000000)
    Ptotal = np.load(directory + "/P-solution.npy", allow_pickle=False)
    times  = np.load(directory + "/times.npy", allow_pickle=False)
    radii  = np.load(directory + "/radii.npy", allow_pickle=False)

    Nr = len(radii)
    dr = radii[1] - radii[0]

    # solve for the TOV solution without any axions at all
    RGsols = solve_for_R_and_G(Rcentral, RHOLS, radii)
    G0vals = RGsols[Nr:]
    R0vals = RGsols[0:Nr]
    F0vals = solve_for_F_TOV(R0vals, G0vals, radii)
    
    # get frequency data from Fourier Transform
    core_frequencies = get_frequencies_1D(Atotal, times, 10, 1000)
    core_frequencies_w = core_frequencies[0]
    core_frequencies_A = core_frequencies[1]
    core_peak_data = get_fft_peaks_and_widths(core_frequencies_w, core_frequencies_A)
    core_peak_ws, core_peak_As, core_fwhms = core_peak_data[0], core_peak_data[1], core_peak_data[2]
    
    crust_frequencies = get_frequencies_1D(Atotal, times, 800, 0)
    crust_frequencies_w = crust_frequencies[0]
    crust_frequencies_A = crust_frequencies[1]
    crust_peak_data = get_fft_peaks_and_widths(crust_frequencies_w, crust_frequencies_A)
    crust_peak_ws, crust_peak_As, crust_fwhms = crust_peak_data[0], crust_peak_data[1], crust_peak_data[2]

    outer_frequencies = get_frequencies_1D(Atotal, times, 2000, 1000)
    outer_frequencies_w = outer_frequencies[0]
    outer_frequencies_A = outer_frequencies[1]
    outer_peak_data = get_fft_peaks_and_widths(outer_frequencies_w, outer_frequencies_A)
    outer_peak_ws, outer_peak_As, outer_fwhms = outer_peak_data[0], outer_peak_data[1], outer_peak_data[2]
    

    # frequency data important
    if core_peak_ws == []:
        core_peak_ws.append(0)
        core_peak_As.append(0)
        core_fwhms.append(0)

    if crust_peak_ws == []:
        crust_peak_ws.append(0)
        crust_peak_As.append(0)
        crust_fwhms.append(0)
    
    if outer_peak_ws == []:
        outer_peak_ws.append(0)
        outer_peak_As.append(0)
        outer_fwhms.append(0)

    important_w_info = [core_peak_ws[0], core_peak_As[0], core_fwhms[0], crust_peak_ws[0], crust_peak_As[0], crust_fwhms[0], outer_peak_ws[0], outer_peak_As[0], outer_fwhms[0]]
    
    #maxafterindex = 100
    #maxouterfrequencyamplitudeindex = np.argmax(outer_frequencies_A[maxafterindex:]) + maxafterindex
    #maxouterfrequency = outer_frequencies_w[maxouterfrequencyamplitudeindex]
    maxouterfrequency = outer_peak_ws[0]

    # calculate the mass in the axion field and calculate the radius of the new star
    # first calculate axion density and baryon density and total density
    baryondensity    = R0vals * (mN0 + sigmaN * (np.sqrt(1 - ((4*mu*md)/(mu+md)**2)*np.sin(Atotal[-1,:])) - 1)) / mN0
    totaldensity     = inside_density(radii, Gtotal[-1, :], R0vals, Atotal[-1,:], Ptotal[-1,:], first_r_derivative(Atotal[-1,:], dr), fa, epsilon)

    # now calculate total mass, and axion mass
    Mtotal = get_total_inside_mass_fast(radii, Gtotal[-1, :], R0vals, Atotal[-1,:], Ptotal[-1,:], first_r_derivative(Atotal[-1,:], dr), fa, epsilon)
    Monlyb = get_total_inside_mass_fast(radii, G0vals, R0vals, np.zeros(np.shape(R0vals)), np.zeros(np.shape(R0vals)), np.zeros(np.shape(R0vals)), 10**50, 0)

    # now calculate the radius with baryons only and then with the total density
    Rtotal = find_radius(totaldensity, radii) 
    Ronlyb = find_radius(R0vals, radii)
    Ronlybbutwithaxions = find_radius(baryondensity, radii)


    # calculate the things you need to get the tidal deformabilities
    Z0 = zeta_perturbation(Ftotal[-1, :], F0vals)
    D0 = delta_perturbation(Gtotal[-1, :], G0vals)
    M0 = mass_TOV(radii, G0vals)
    Ra0 = axion_density(radii, Gtotal[-1, :], R0vals, Atotal[-1, :], np.zeros(np.shape(radii)), first_r_derivative(Atotal[-1,:], dr), fa, epsilon)
    Pa0 = axion_radial_pressure(radii, Gtotal[-1, :], R0vals, Atotal[-1, :], np.zeros(np.shape(radii)), first_r_derivative(Atotal[-1,:], dr), fa, epsilon)
    dRdr0 = first_r_derivative(R0vals, dr)
    dRadr0 = first_r_derivative(Ra0, dr)
    Mbaryon = Monlyb
    Maxion = Mtotal - Monlyb
    
    tidal_deformabilities = solve_tidal_deformabilities(radii, Z0, D0, M0, R0vals, Ra0, Pa0, dRdr0, dRadr0, Mbaryon, Maxion)
    
    # set up returns
    # return core requencies and amplitudes, then 
    del(Ftotal, Gtotal, Atotal, Ptotal)

    return [core_frequencies_w, core_frequencies_A, crust_frequencies_w, crust_frequencies_A, outer_frequencies_w, outer_frequencies_A, [epsilon, Rcentral, ma, fa, maxouterfrequency, Ronlyb, Ronlybbutwithaxions, Rtotal, Monlyb, Maxion, Mtotal, tidal_deformabilities[0], tidal_deformabilities[1], tidal_deformabilities[0] + tidal_deformabilities[1]], important_w_info];



########################## DO THE ANALYSIS ###########################
masses = []
fas = []
frequencies = []
epsilons = []
Rcentrals = []

total_data = []

#okayindices = list(range(888))

startindex = 601

for i in list(range(startindex, 888)):
    directory = "/scratch/wentzel4/axions-in-NS/SLy4-simulation/outputs-npy/SIM_" + str(i) + "/"
    currentdata = total_analysis_function(directory, i)
    
    print("columns are: epsilon, Rcentral, ma, fa, woutmax, Rs with no axions, Rs with only baryons in presence of axion, Rs total, Ms with no axions, mass in axion field, Ms with all mass, Lambda baryon, Lambda axion, total Lambda, wcoremax, wcoremaxamp, wcoremaxfwhm, same for crust, same for outer")
    print(np.concatenate((currentdata[6], currentdata[7])))
    total_data.append(np.concatenate((currentdata[6], currentdata[7])))
    
    np.savetxt("/scratch/wentzel4/axions-in-NS/SLy4-simulation/outputs-npy/analysis/simulation-analysis-" + str(startindex) + ".txt", total_data, fmt='%.18e', delimiter=',', header="analysis for parameter sets " + str(startindex) + " to " + str(i)  + ". Columns are: epsilon, Rcentral, ma, fa, woutmax, Rs with no axions, Rs with only baryons in presence of axion, Rs total, Ms with no axions, mass in axion field, Ms with all mass, Lambda baryon, Lambda axion, total Lambda, wcoremax, wcoremaxamp, wcoremaxfwhm, same for crust, same for outer")
    print("done with file " + str(i) + " of 888")
    cmd = "echo {}".format(str("Done with file " + str(i) + " of 888"))
    os.system(cmd)

np.savetxt("/scratch/wentzel4/axions-in-NS/SLy4-simulation/outputs-npy/analysis/simulation-analysis.txt", total_data, fmt='%.18e', delimiter=',', header="columns are: epsilon, Rcentral, ma, fa, woutmax, Rs with no axions, Rs with only baryons in presence of axion, Rs total, Ms with no axions, mass in axion field, Ms with all mass, Lambda baryon, Lambda axion, total Lambda, wcoremax, wcoremaxamp, wcoremaxfwhm, same for crust, same for outer")

print("Saved data")


print("Saved data. Goodbye")
