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



def calculate_RMLC_from_file(PARAMETERINDEX):
    DATADIRECTORY         = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/SIM_restest" + str(PARAMETERINDEX)
    PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"

    PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")
    epsilon      = PARAMETERSET[PARAMETERINDEX, 2]
    fa           = PARAMETERSET[PARAMETERINDEX, 1]
    nNcentral    = PARAMETERSET[PARAMETERINDEX, 0]

    RESOLUTION = 20

    nNCUT = 1e-20*nsatinGeV3
    nNRadiusCUT = 1e-5*nsatinGeV3

    ma = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)

    rvals = np.load(DATADIRECTORY + "/radii.npy")
    tvals = np.load(DATADIRECTORY + "/times.npy")
    tvals = tvals[::RESOLUTION]

    Nr = len(rvals)

    GNTOV = solve_for_G_N_noaxion(nNcentral, nNCUT, rvals)
    GTOV, nNTOV = GNTOV[:Nr], GNTOV[Nr:]

    RMCLTOV = calculate_gravitational_observables(rvals, GTOV, nNTOV, np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), epsilon, fa, nNRadiusCUT)

    try:
        Fvals = np.load(DATADIRECTORY + "/F-solution.npy")
        Fvals = np.load(DATADIRECTORY + "/F-solution.npy")
        Gvals = np.load(DATADIRECTORY + "/G-solution.npy")
        nNvals = np.load(DATADIRECTORY + "/nN-solution.npy")
        Uvals = np.load(DATADIRECTORY + "/U-solution.npy")
        Avals = np.load(DATADIRECTORY + "/A-solution.npy")
        Pvals = np.load(DATADIRECTORY + "/P-solution.npy")
        MNSvals = np.load(DATADIRECTORY + "/NSmasses.npy")
        RNSvals = np.load(DATADIRECTORY + "/NSradii.npy")
        print(f"Successfully processed {PARAMETERINDEX}")
    except Exception as e:
        print(f"Error processing {PARAMETERINDEX}: {e}")
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # calculate kinetic energies and dRdt
    dRdt = (np.roll(RNSvals, -1, 0) - RNSvals) / (tvals[1] - tvals[0])
    KEvals = np.zeros(np.shape(Uvals))
    KETotalVals = np.zeros(np.shape(KEvals)[0])
    for i in range(int(len(RNSvals)/RESOLUTION)):
        KEvals[i,:] = kinetic_energy_density(rvals, Gvals[i,:], nNvals[i,:], Uvals[i,:], Avals[i,:], Pvals[i,:], epsilon, fa)
        KEvals[i,:] = KEvals[i,:] * 4*np.pi*rvals**2 * (rvals[1] - rvals[0]) * Gvals[i,:] * Fvals[i,:]
        KETotalVals[i] = np.sum(KEvals[i,:np.argmin(np.abs(rvals - np.ones(np.shape(rvals))*RNSvals[i*RESOLUTION]))])
    
    print("Approximate maximum kinetic energy fraction: " + str(np.max(KETotalVals / MNSvals[1])))
    
    #final_index = int(len(MNSvals) / RESOLUTION - 1)
    # starting value for the cutoff in time needs to be larger than 1 to account for weird time derivatives in the first step
    #final_index = 1
    # KETotalVals[final_index] > 0.001 * (np.max(KETotalVals / MNSvals[1])) * MNSvals[final_index*RESOLUTION]
    #while (final_index*RESOLUTION < len(dRdt) and KETotalVals[final_index] > 0.01 * (np.max(KETotalVals / MNSvals[1])) * MNSvals[final_index*RESOLUTION] and MNSvals[final_index*RESOLUTION] > 0.90 * RMCLTOV[1]) or (final_index*RESOLUTION < len(dRdt) and abs(dRdt[final_index*RESOLUTION]) > 0.0):
     #   final_index += 1
    try:
        final_index = np.min(zero_crossings_indices(KETotalVals - np.ones(np.shape(KETotalVals)) * 0.001 * np.max(KETotalVals), min_index=50))
    except Exception as e:
        final_index = int(len(RNSvals) / RESOLUTION) - 1
        print("minimum kinetic energy never reached: {e}")

    print("final index: " + str(final_index))

    if abs(dRdt[final_index*RESOLUTION]) > 0.0:
        final_index = np.min(np.where(dRdt[final_index:] != 0.0)) + final_index 

    print("length of dRdt is: " + str(len(dRdt)))
    print("Final index obtained: " + str(final_index) + "   ---- should be larger than " + str(int(len(dRdt) / RESOLUTION / 4)))
    
    if abs(dRdt[final_index*RESOLUTION]) > 0.0:
        print("radius not fully stabilized, dRdt * Deltat = " + str(dRdt[final_index*RESOLUTION] * (tvals[1] - tvals[0]) * GeV_2_km))
    #for i in range(len(KETotalVals))
    #for i in range(len(MNSvals)):
    #    if MNSvals[i] > 0.95 * RMCLTOV[1]:
    #        final_index = i

    final_index = int(final_index - 1)

    # calculate the mass and radius for your solution and the TOV solution
    Fv, Gv, nNv, Uv, Av, Pv = Fvals[final_index,:], Gvals[final_index,:], nNvals[final_index,:], Uvals[final_index,:], Avals[final_index,:], Pvals[final_index,:]
    RMCLaxion = calculate_gravitational_observables(rvals, Gv, nNv, np.zeros(np.shape(rvals)), Av, np.zeros(np.shape(rvals)), epsilon, fa, nNRadiusCUT)

    # speed of sound calculations
    total_pressure_values = smooth(total_pressure(rvals, Gvals[final_index,:], nNvals[final_index,:], Uvals[final_index,:], Avals[final_index,:], Pvals[final_index,:], epsilon, fa), 1) 
    total_pressure_values = enforce_rc_BC(total_pressure_values, ZEROATRC=1)
    total_density_values = smooth(total_density(rvals, Gvals[final_index,:], nNvals[final_index,:], Uvals[final_index,:], Avals[final_index,:], Pvals[final_index,:], epsilon, fa), 1)
    total_density_values  = enforce_rc_BC(total_density_values, ZEROATRC=1)
    dPdr_vals   = first_r_derivative(total_pressure_values, rvals[1] - rvals[0])
    dRhodr_vals = first_r_derivative(total_density_values, rvals[1] - rvals[0])
    dPdRho_vals = np.zeros(np.shape(dPdr_vals))
    np.divide(dPdr_vals, dRhodr_vals, out=dPdRho_vals, where=(np.abs(dRhodr_vals)>0.0))

    speed_of_sound_at_center = dPdRho_vals[1]
    speed_of_sound_at_R = dPdRho_vals[int(np.argmin(np.abs(rvals - np.ones(np.shape(rvals))*RMCLaxion[0]))-1)]
    
    baryon_pressure_values = baryon_pressure(rvals, GTOV, nNTOV, np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), epsilon, fa)
    baryon_pressure_values = enforce_rc_BC(baryon_pressure_values, ZEROATRC=1)
    baryon_density_values = baryon_density(rvals, GTOV, nNTOV, np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), np.zeros(np.shape(rvals)), epsilon, fa)
    baryon_density_values  = enforce_rc_BC(baryon_density_values, ZEROATRC=1)
    dPdr_vals_noaxion   = first_r_derivative(baryon_pressure_values, rvals[1] - rvals[0])
    dRhodr_vals_noaxion = first_r_derivative(baryon_density_values, rvals[1] - rvals[0])
    dPdRho_vals_noaxion = np.zeros(np.shape(dPdr_vals_noaxion))
    np.divide(dPdr_vals_noaxion, dRhodr_vals_noaxion, out=dPdRho_vals_noaxion, where=(np.abs(dRhodr_vals_noaxion)>0.0))

    speed_of_sound_at_center_noaxion = dPdRho_vals_noaxion[1]
    speed_of_sound_at_R_noaxion = dPdRho_vals_noaxion[int(np.argmin(np.abs(rvals - np.ones(np.shape(rvals))*RMCLTOV[0]))-1)]
    
    print("initial, final radius: " + str(RMCLTOV[0] * GeV_2_km) + "    " + str(RMCLaxion[0] * GeV_2_km) + " km")
    print("initial, final mass:   " + str(RMCLTOV[1] * GeV_2_kg * kg_2_Msun) + "    " + str(RMCLaxion[1] * GeV_2_kg * kg_2_Msun) + " km")
    print("initial, final lambda: " + str(RMCLTOV[3]) + "    " + str(RMCLaxion[3]) + " km")
    print("cs noaxion, cs axion: " + str(speed_of_sound_at_center_noaxion) + "     " + str(speed_of_sound_at_center))
    print("total time steps: " + str(len(MNSvals)))

    return np.array([RMCLTOV[0], RMCLTOV[1], RMCLTOV[2], RMCLTOV[3], RMCLaxion[0], RMCLaxion[1], RMCLaxion[2], RMCLaxion[3], speed_of_sound_at_center, speed_of_sound_at_center_noaxion, speed_of_sound_at_R, speed_of_sound_at_R_noaxion]);



