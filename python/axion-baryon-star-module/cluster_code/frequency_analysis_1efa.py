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

RESOLUTION = 20

indices = [27, 28, 31, 45, 46, 49, 90, 91, 94, 180, 184, 181, 270, 271, 274, 360, 361, 364, 450, 454, 451, 540, 544, 541, 630, 631, 634, 720, 721, 724, 810, 811, 814]
total_fs_data = [] 
total_As_data = []

for index in indices:
    try:
        directory = DATADIRECTORY + "SIM_" + str(index)
        NSradii = np.load(directory + "/NSradii.npy")
        RNS = NSradii[-5]
        frequency_data = total_frequency_analysis_function(directory)
        fs_nucleon = frequency_data[2]
        As_nucleon = frequency_data[3]
        fs_nucleon_all = frequency_data[6]
        As_nucleon_all = frequency_data[7]
        fs_KE_all = frequency_data[8]
        As_KE_all = frequency_data[9]
        #fs_jr = frequency_data[2]
        #As_jr = frequency_data[3]

        epsilon      = PARAMETERSET[index, 2]
        fa           = PARAMETERSET[index, 1]
        nNcentral    = PARAMETERSET[index, 0]
        ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (2*fa**2)) * np.sqrt(mu*md / (mu + md)**2)
#        print('final radius: ' + str(NSradii[9500]))
        
        np.save(OUTPUTDIRECTORY + "nucleon_frequency_data_" + str(index) + ".npy", np.array(fs_nucleon_all))
        np.save(OUTPUTDIRECTORY + "nucleon_amplitude_data_" + str(index) + ".npy", np.array(As_nucleon_all))
        np.save(OUTPUTDIRECTORY + "nucleon_frequency_peaks_" + str(index) + ".npy", np.array(fs_nucleon))
        np.save(OUTPUTDIRECTORY + "nucleon_amplitude_peaks_" + str(index) + ".npy", np.array(As_nucleon))
        np.save(OUTPUTDIRECTORY + "nucleon_frequency_KE_data_" + str(index) + ".npy", np.array(fs_KE_all))
        np.save(OUTPUTDIRECTORY + "nucleon_amplitude_KE_data_" + str(index) + ".npy", np.array(As_KE_all))

        print("done with file " + str(index), flush=True)
        #print("axion frequency: " + str(famax) + " rad/s      axion mass: " + str(ma) + "       nucleon frequency: " + str(fjmax) + " rad/s", flush=True)

    except Exception as e:
        print("error processing file " + str(index), flush=True)
        print(e)
