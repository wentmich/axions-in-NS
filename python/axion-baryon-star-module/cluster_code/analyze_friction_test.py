# analyze friction independence
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

PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets_fric_test_fine.csv"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")

indices = range(10)
final_axion_data = np.zeros((len(indices), 10000))
# final_energy_data = np.zeros((len(indices), 10000))
# final_pressure_data = np.zeros((len(indices), 10000))
final_nN_data = np.zeros((len(indices), 10000))
initial_axion_data = np.zeros(10000)

final_masses = np.zeros(len(indices))
final_radii  = np.zeros(len(indices))

for index in indices:
    DATADIRECTORY         = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/SIM_frictest8_" + str(index)

    rvals = np.load(DATADIRECTORY + "/radii.npy")
    tvals = np.load(DATADIRECTORY + "/times.npy")
    tvals = tvals[::RESOLUTION]

    try:
        nNvals = np.load(DATADIRECTORY + "/nN-solution.npy")
        Avals = np.load(DATADIRECTORY + "/A-solution.npy")
        Gvals = np.load(DATADIRECTORY + "/G-solution.npy")
        Fvals = np.load(DATADIRECTORY + "/F-solution.npy")
        Uvals = np.load(DATADIRECTORY + "/U-solution.npy")
        MNSvals = np.load(DATADIRECTORY + "/NSmasses.npy")
        RNSvals = np.load(DATADIRECTORY + "/NSradii.npy")
        print(f"Successfully processed file {index}")
    except Exception as e:
        print(f"Error processing file {index}: {e}")
    
    final_index = int(len(RNSvals)/RESOLUTION)

    rout = -1
    initial_baryon_number = np.sum(4*np.pi*rvals[:rout]**2*(rvals[1] - rvals[0])*Gvals[0,:rout]  *nNvals[0,:rout]/(Fvals[0,:rout]*np.sqrt(1 - Uvals[0,:rout]**2)))
    final_baryon_number = np.sum(4*np.pi*rvals[:rout]**2*(rvals[1] - rvals[0])*Gvals[final_index,:rout] *nNvals[final_index,:rout] / (Fvals[0,:rout]*np.sqrt(1 - Uvals[final_index,:rout])))
    print(final_baryon_number / initial_baryon_number)
    
    final_axion_data[index, :] = Avals[final_index, :]
    final_nN_data[index, :] = nNvals[final_index, :]
    initial_axion_data = Avals[0, :]
    final_masses[index] = MNSvals[final_index*RESOLUTION]
    final_radii[index] = RNSvals[final_index*RESOLUTION]

    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/friction_test_8_final_Avals.npy", final_axion_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/friction_test_8_initial_Avals.npy", initial_axion_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/friction_test_8_final_nNvals.npy", final_nN_data)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/friction_test_8_final_Mvals.npy", final_masses)
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/friction_test_8_final_Rvals.npy", final_radii)

    
