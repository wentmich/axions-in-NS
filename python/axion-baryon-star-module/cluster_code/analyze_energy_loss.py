# check energy loss rates
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

STARTINDEX        = int(sys.argv[1])
epsexp = int(sys.argv[2])
faexp  = int(sys.argv[3])

PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")


indices = range(int(STARTINDEX), 899, 9)
total_data = np.zeros((len(indices), 7))

for j in range(len(indices)):
    index = indices[j]

    DATADIRECTORY         = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/SIM_fric_" + str(index)

    epsilon      = PARAMETERSET[index, 2]
    fa           = PARAMETERSET[index, 1]
    nNcentral    = PARAMETERSET[index, 0]
    ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)

    rvals = np.load(DATADIRECTORY + "/radii.npy")
    tvals = np.load(DATADIRECTORY + "/times.npy")
    tvals = tvals[::RESOLUTION]

    BNDINDEX = 8000

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
        print(f"Successfully processed file {index}")
    except Exception as e:
        print(f"Error processing file {index}: {e}")
        total_data[index, :] = np.zeros(7)

    final_index = int(len(RNSvals)/RESOLUTION - 1)
    total_energy_initial = np.sum(4*np.pi*rvals[:BNDINDEX]**2*(rvals[1] - rvals[0])*total_density(rvals, Gvals[0, :], nNvals[0, :], Uvals[0, :], Avals[0, :], Pvals[0, :], epsilon, fa)[:BNDINDEX])
    total_energy_final   = np.sum(4*np.pi*rvals[:BNDINDEX]**2*(rvals[1] - rvals[0])*total_density(rvals, Gvals[final_index, :], nNvals[final_index, :], Uvals[final_index, :], Avals[final_index, :], Pvals[final_index, :], epsilon, fa)[:BNDINDEX])

#    total_energy_flux = np.sum(4*np.pi*rvals[BNDINDEX]**2 * (tvals[1] - tvals[0])*RESOLUTION * total_radial_energy_flux(rvals, Fvals, Gvals, nNvals, Uvals, Avals, Pvals, fa)[:final_index, BNDINDEX])
    
    total_energy_flux = 0.0
    for i in range(final_index):
        total_energy_flux = total_energy_flux + total_radial_energy_flux(rvals, Fvals[i,:], Gvals[i,:], nNvals[i,:], Uvals[i,:], Avals[i,:], Pvals[i,:], fa)[BNDINDEX]

    total_energy_flux = -total_energy_flux * 4*np.pi*rvals[BNDINDEX]**2 * (tvals[1] - tvals[0])

    total_data[j,:] = np.array([nNcentral, epsilon, fa, ma, total_energy_initial, total_energy_final, total_energy_flux])
    
    print("Total energy loss fraction: " + str((total_energy_initial - total_energy_final) / total_energy_initial) + "        total energy flux: " + str(total_energy_flux / total_energy_initial))
    print("done with file " + str(index))
    
    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/energy-loss-rates-eps1e-" + str(epsexp) + "_fa1e" + str(faexp) + ".npy", total_data)
