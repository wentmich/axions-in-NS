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
from calculate_gravitational_observables import *

STARTINDEX        = int(sys.argv[1])
epsexp = int(sys.argv[2])
faexp  = int(sys.argv[3])

PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")


indices = range(int(STARTINDEX), 899, 9)
total_data = np.zeros((len(indices), 16))

for index in indices:
    try:
        mydata = calculate_RMLC_from_file(index)
    except Exception as e:
        print("error at index: " + str(index) + "   " + str(e))    
        mydata = np.zeros(12)

    epsilon      = PARAMETERSET[index, 2]
    fa           = PARAMETERSET[index, 1]
    nNcentral    = PARAMETERSET[index, 0]
    ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)
    
    total_data[int(index / 9), :] = np.concatenate((np.array([nNcentral, epsilon, fa, ma]), mydata*np.array([GeV_2_km, GeV_2_kg * kg_2_Msun, GNnat, 1, GeV_2_km, GeV_2_kg * kg_2_Msun, GNnat, 1.0, 1.0, 1.0, 1.0, 1.0])))

    np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/MRCL_KEcut_eps1e-" + str(epsexp) + "_fa1e" + str(faexp) + ".npy", total_data)

    print("Done with file " + str(int(index/9)) + " of " + str(len(indices)), flush=True)

