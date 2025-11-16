# analyze the frequency of axion and baryon oscillations
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
from frequency_peak_functions import *


PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"
DATADIRECTORY = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/"
OUTPUTDIRECTORY = "/u/wentzel4/axions-in-NS/total-dynamics/outputs/"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")

RESOLUTION = 1

indices = range(396, 405)
total_data = np.zeros((len(indices), 7)) 

for index in indices:
    try:
        directory = DATADIRECTORY + "SIM_hires_" + str(index)
        NSradii = np.load(directory + "/NSradii.npy")
        RNS = NSradii[-5]
        frequency_data = total_frequency_analysis_function(directory)
        fs_axion = frequency_data[0]
        As_axion = frequency_data[1]
        fs_jr = frequency_data[2]
        As_jr = frequency_data[3]

        epsilon      = PARAMETERSET[index, 2]
        fa           = PARAMETERSET[index, 1]
        nNcentral    = PARAMETERSET[index, 0]
        ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)
        
        famax = fs_axion[np.argmax(np.array(As_axion))]
        fjmax = fs_jr[np.argmax(np.array(As_jr))]

        total_data[index, :] = np.array([nNcentral, epsilon, fa, ma, famax, fjmax, RNS])

        np.save(OUTPUTDIRECTORY + "peak_frequency_data_hires.npy", total_data)

        print("done with file " + str(index), flush=True)
        print("axion frequency: " + str(famax) + " rad/s      axion mass: " + str(ma) + "       nucleon frequency: " + str(fjmax) + " rad/s", flush=True)

    except Exception as e:
        print("error processing file " + str(index), flush=True)
        print(e)
