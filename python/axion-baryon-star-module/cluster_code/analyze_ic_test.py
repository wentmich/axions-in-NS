# analyze ic independence
import sys
import numpy as np
from numpy import cos, sin, tan, roll
from auxilliary_functions import *
from constants_GeV import *
from EOS import *
from TOV_initial_state import *
from metric_solver_radial_RK4 import *
from matter_and_metric_solver import *
from gravitational_observables import *
from density_pressure_functions import *
from save_data import *
from calculate_gravitational_observables import *

RESOLUTION = 20

PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets_ic_test.csv"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")

indices = range(25)
final_axion_data = np.zeros((len(indices), 10000))
# final_energy_data = np.zeros((len(indices), 10000))
# final_pressure_data = np.zeros((len(indices), 10000))
final_nN_data = np.zeros((len(indices), 10000))
initial_axion_data = np.zeros((len(indices), 10000))

final_masses = np.zeros(len(indices))
final_radii  = np.zeros(len(indices))

for index in indices:
    DATADIRECTORY         = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/SIM_ictest_" + str(index)

    rvals = np.load(DATADIRECTORY + "/radii.npy")
    tvals = np.load(DATADIRECTORY + "/times.npy")
    tvals = tvals[::RESOLUTION]
    """
    try:
        nNvals = np.load(DATADIRECTORY + "/nN-solution.npy")
        Avals = np.load(DATADIRECTORY + "/A-solution.npy")
        MNSvals = np.load(DATADIRECTORY + "/NSmasses.npy")
        RNSvals = np.load(DATADIRECTORY + "/NSradii.npy")
        print(f"Successfully processed file {index}")
    except Exception as e:
        print(f"Error processing file {index}: {e}")
    
    final_index = int(len(RNSvals)/RESOLUTION)
    """;
    #######################
    try:
        Gvals = np.load(DATADIRECTORY + "/G-solution.npy")
        nNvals = np.load(DATADIRECTORY + "/nN-solution.npy")
        Uvals = np.load(DATADIRECTORY + "/U-solution.npy")
        Avals = np.load(DATADIRECTORY + "/A-solution.npy")
        Pvals = np.load(DATADIRECTORY + "/P-solution.npy")
        MNSvals = np.load(DATADIRECTORY + "/NSmasses.npy")
        RNSvals = np.load(DATADIRECTORY + "/NSradii.npy")
        print(f"Successfully processed {index}")
    except Exception as e:
        print(f"Error processing file {index}: {e}")

    # calculate kinetic energies and dRdt
    dRdt = (np.roll(RNSvals, -1, 0) - RNSvals) / (tvals[1] - tvals[0])
    KEvals = np.zeros(np.shape(Uvals))
    KETotalVals = np.zeros(np.shape(KEvals)[0])
    for i in range(int(len(RNSvals)/RESOLUTION)):
        KEvals[i,:] = kinetic_energy_density(rvals, Gvals[i,:], nNvals[i,:], Uvals[i,:], Avals[i,:], Pvals[i,:], 1e-3, 1e15) 
        KEvals[i,:] = KEvals[i,:] * 4*np.pi*rvals**2 * (rvals[1] - rvals[0])
        KETotalVals[i] = np.sum(KEvals[i,:np.argmin(np.abs(rvals - np.ones(np.shape(rvals))*RNSvals[i*RESOLUTION]))])
    
    final_index = np.min(zero_crossings_indices(KETotalVals - np.ones(np.shape(KETotalVals)) * 0.01 * np.max(KETotalVals), min_index=50))
    print("final index: " + str(final_index))
    print("dRdt at final: " + str(dRdt[final_index*20]))
 
    ########################
    final_axion_data[index, :] = Avals[final_index, :]
    final_nN_data[index, :] = nNvals[final_index, :]
    initial_axion_data[index, :] = Avals[0, :]
    final_masses[index] = MNSvals[final_index*RESOLUTION]
    final_radii[index] = RNSvals[final_index*RESOLUTION]

    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/ic_test_2_final_Avals.npy", final_axion_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/ic_test_2_initial_Avals.npy", initial_axion_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/ic_test_2_final_nNvals.npy", final_nN_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/ic_test_2_final_Mvals.npy", final_masses)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/ic_test_2_final_Rvals.npy", final_radii)

    
