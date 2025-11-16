import sys
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import time as time
import scipy.ndimage as scimage
import math
from numpy import cos, sin, tan, roll
from scipy.interpolate import UnivariateSpline
from auxilliary_functions import *
from constants_GeV import *
from EOS import *
from TOV_initial_state import *
from metric_solver_radial_RK4 import *
from matter_and_metric_solver import *
from gravitational_observables import *
from density_pressure_functions import *
from save_data import *


PARAMETERINDEX        = int(sys.argv[1])
DATADIRECTORY         = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/SIM_drtest_" + str(PARAMETERINDEX)
PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"


PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")
epsilon      = PARAMETERSET[PARAMETERINDEX, 2]
fa           = PARAMETERSET[PARAMETERINDEX, 1]
nNcentral    = PARAMETERSET[PARAMETERINDEX, 0]

Nr = 20000
dr = 5.0 * 5.06773e+15
rinit, rfin = dr/2, (Nr * dr + dr/2)
rvals = np.linspace(rinit, rfin, Nr)

Nt = 40000
dt = dr/1.0
tinit, tfin = 0.0, Nt*dt
tvals = np.linspace(tinit, tfin, Nt)

RESOLUTION = 20

nNCUT = 1e-20*nsatinGeV3
nNRadiusCUT = 1e-5*nsatinGeV3

ma = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)

Uinit = np.zeros(np.shape(rvals))
Ainit = np.zeros(np.shape(rvals))
Pinit = np.zeros(np.shape(rvals))

GNTOV = solve_for_G_N_noaxion(nNcentral, nNCUT, rvals)
GTOV, nNinit = GNTOV[:Nr], GNTOV[Nr:]

# smoothed step function for the axion
i = 0
pressures = total_pressure(rvals, GTOV, nNinit, Uinit, np.ones(np.shape(rvals))*np.pi, Pinit, epsilon, fa)
while pressures[i] > 0.0 and i < Nr - 1:
    Ainit[i] = np.pi
    i = i+1

#### CHANGED FOR DIFFERENT RADIAL RESOLTUION
Ainit = smooth(Ainit, int(Nr/100))

# gaussian profile for the axion
#std = AXIONSTD
#Ainit = np.exp(-(rvals)**2 / (2.0*std**2)) / (np.sqrt(2.0*np.pi) * std)
#Ainit = np.pi * Ainit / Ainit[0]

Ginit = solve_G_constraint_fast(rvals, nNinit, Uinit, Ainit, Pinit, epsilon, fa)
Finit = solve_F_constraint_fast(rvals, Ginit, nNinit, Uinit, Ainit, Pinit, epsilon, fa)




##### EVOLUTION #####
# evolve system
NSMassVals = []
NSRadiusVals = []
NumberOfBaryons = []

FTotalvals, GTotalvals, nNTotalvals, UTotalvals, ATotalvals, PTotalvals = np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr))
FTotalvals[0, :] = Finit
GTotalvals[0, :] = Ginit
nNTotalvals[0, :] = nNinit
UTotalvals[0, :] = Uinit
ATotalvals[0, :] = Ainit
PTotalvals[0, :] = Pinit

Fv = np.copy(Finit)
Gv = np.copy(Ginit)
nNv = np.copy(nNinit)
Uv = np.copy(Uinit)
Av = np.copy(Ainit)
Pv = np.copy(Pinit)

total_non_causal = 0

for i in range(Nt):
    # maybe replace nNCUT in the line below with nNRadiusCUT
    newvals = matter_and_metric_solver(rvals, Fv, Gv, nNv, Uv, Av, Pv, epsilon, fa, dt, nNRadiusCUT)
    Fv, Gv, nNv, Uv, Av, Pv = newvals[:Nr], newvals[Nr:2*Nr], newvals[2*Nr:3*Nr], newvals[3*Nr:4*Nr], newvals[4*Nr:5*Nr], newvals[5*Nr:6*Nr]

    if np.any(Uv > 0.9) or np.any(Uv < -0.9):
        total_non_causal += 1

    Uv[Uv > 0.8] = 0.8
    Uv[Uv < -0.8] = -0.8

    mass_and_radius = get_NS_mass_and_radius_from_G0(rvals, nNv, nNRadiusCUT, Gv)
    NSMassVals.append(mass_and_radius[1])
    NSRadiusVals.append(mass_and_radius[0])
    NumberOfBaryons.append(np.sum(nNv * 4 * np.pi * rvals**2 * dr * Gv / np.sqrt(1 - Uv**2)))
    
    if i%RESOLUTION == 0:
        FTotalvals[int(i/RESOLUTION),:] = Fv
        GTotalvals[int(i/RESOLUTION),:] = Gv
        nNTotalvals[int(i/RESOLUTION),:] = nNv
        UTotalvals[int(i/RESOLUTION),:] = Uv
        ATotalvals[int(i/RESOLUTION),:] = Av
        PTotalvals[int(i/RESOLUTION),:] = Pv
        print("Done with step " + str(i) + " of " + str(Nt), flush=True)
        print("Total steps with u > 0.8: " + str(total_non_causal))
        print("Current Mass and Radius: " + str(NSMassVals[-1]*GeV_2_kg*kg_2_Msun) + " Msun,  " + str(NSRadiusVals[-1] / 5.06773e+18) + " km", flush=True)
        print("Currrent Baryon Number: " + str(NumberOfBaryons[-1]), flush=True)
        
    if i%(RESOLUTION) == 0:
        save_total_data_binary(FTotalvals, GTotalvals, nNTotalvals, UTotalvals, ATotalvals, PTotalvals, tvals, rvals, np.array(NSMassVals), np.array(NSRadiusVals), [nNcentral, fa, ma, epsilon], DATADIRECTORY)
        
    if np.isnan(Fv).any() or np.isnan(Gv).any() or np.isnan(nNv).any() or np.isnan(Uv).any() or np.isnan(Av).any() or np.isnan(Pv).any():
        if np.isnan(Fv).any():
            print("F")
            print(np.argwhere(np.isnan(Fv)))
        if np.isnan(Gv).any():
            print("G")
            print(np.argwhere(np.isnan(Gv)))
        if np.isnan(nNv).any():
            print("n")
            print(np.argwhere(np.isnan(nNv)))
        if np.isnan(Uv).any():
            print("U")
            print(np.argwhere(np.isnan(Uv)))
        if np.isnan(Av).any():
            print("A")
            print(np.argwhere(np.isnan(Av)))
        if np.isnan(Pv).any():
            print("P")
            print(np.argwhere(np.isnan(Pv)))
        break;







