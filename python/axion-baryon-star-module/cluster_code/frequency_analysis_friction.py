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


PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets_fric_test_fine.csv"
DATADIRECTORY = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/"
OUTPUTDIRECTORY = "/u/wentzel4/axions-in-NS/total-dynamics/outputs/"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")

RESOLUTION = 20

indices = [0,1,2,3,4,5,6,7,8,9]
total_fs_data = [] 
total_As_data = []

for index in indices:
    try:
        directory = DATADIRECTORY + "SIM_frictest8_" + str(index)
        NSradii = np.load(directory + "/NSradii.npy")
        RNS = NSradii[-5]
        frequency_data = total_frequency_analysis_function(directory)
        fs_axion = frequency_data[0]
        As_axion = frequency_data[1]
        fs_axion_all = frequency_data[4]
        As_axion_all = frequency_data[5]
        fs_KE_all = frequency_data[6]
        As_KE_all = frequency_data[7]
        #fs_jr = frequency_data[2]
        #As_jr = frequency_data[3]

        epsilon      = PARAMETERSET[index, 2]
        fa           = PARAMETERSET[index, 1]
        nNcentral    = PARAMETERSET[index, 0]
        ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)
#        print('final radius: ' + str(NSradii[9500]))
        
        np.save(OUTPUTDIRECTORY + "friction_frequency_data_" + str(index) + ".npy", np.array(fs_axion_all))
        np.save(OUTPUTDIRECTORY + "friction_amplitude_data_" + str(index) + ".npy", np.array(As_axion_all))
        np.save(OUTPUTDIRECTORY + "friction_frequency_peaks_" + str(index) + ".npy", np.array(fs_axion))
        np.save(OUTPUTDIRECTORY + "friction_amplitude_peaks_" + str(index) + ".npy", np.array(As_axion))

        np.save(OUTPUTDIRECTORY + "friction_frequency_KE_data_" + str(index) + ".npy", np.array(fs_KE_all))
        np.save(OUTPUTDIRECTORY + "friction_amplitude_KE_data_" + str(index) + ".npy", np.array(As_KE_all))

        #np.save(OUTPUTDIRECTORY + "friction_frequency_peaks_" + str(index) + ".npy", np.array(fs_axion))
        #np.save(OUTPUTDIRECTORY + "friction_amplitude_peaks_" + str(index) + ".npy", np.array(As_axion))

        print("done with file " + str(index), flush=True)
        #print("axion frequency: " + str(famax) + " rad/s      axion mass: " + str(ma) + "       nucleon frequency: " + str(fjmax) + " rad/s", flush=True)

    except Exception as e:
        print("error processing file " + str(index), flush=True)
        print(e)
