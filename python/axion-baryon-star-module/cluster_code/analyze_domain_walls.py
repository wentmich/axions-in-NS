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

#def domain_wall_profile(r, r0, delta1):
#    ans = np.pi * (-np.tanh((r - np.ones(np.shape(r))*r0) / delta1) + 1) / 2
#    return ans;

def domain_wall_profile(r, Amp, r0, delta1):
    ansin = (Amp/r0) * np.heaviside(r0 - r, 0)
    ansout = Amp*(np.exp(-(r-r0)/delta1)/(r))*np.heaviside(r - r0, 1)
    ans = ansin + ansout
    return ans;

def fit_domain_wall(rvals, avals):
    try:
        popt, pcov = opt.curve_fit(domain_wall_profile, rvals, avals, p0=np.array([8*5.06773e+18*np.pi, 8*5.06773e+18, 1*5.06773e+18]))
    except Exception as e:
        print(e)
        print("couldn't optimize")
        return np.array([0.0,0.0,0.0])
    return [popt[0], popt[1], popt[2], pcov[1,1], pcov[2,2]];

PARAMETERSETSFILE     = "/u/wentzel4/axions-in-NS/total-dynamics/inputs/simulation_parameter_sets.csv"
DATADIRECTORY = "/scratch/wentzel4/axions-in-NS/total-dynamics/outputs/"
OUTPUTDIRECTORY = "/u/wentzel4/axions-in-NS/total-dynamics/outputs/"

PARAMETERSET = np.loadtxt(PARAMETERSETSFILE, dtype=float, delimiter=",")

RESOLUTION = 20

indices = range(396, 405)
total_data = np.zeros((len(indices), 7))

for index in indices:
    try:
        rvals = np.load(DATADIRECTORY + "SIM_fric_" + str(index) + "/radii.npy")
        avals = np.load(DATADIRECTORY + "SIM_fric_" + str(index) + "/A-solution.npy")
        NSmasses = np.load(DATADIRECTORY + "SIM_fric_" + str(index) + "/NSmasses.npy")

        avals = avals[int(len(NSmasses)/RESOLUTION - 5), :]

        r0delta = fit_domain_wall(rvals, avals)

        r0 = r0delta[1]
        delta1 = r0delta[2]
        delta1cov = r0delta[4]

        epsilon      = PARAMETERSET[index, 2]
        fa           = PARAMETERSET[index, 1]
        nNcentral    = PARAMETERSET[index, 0]
        ma           = np.sqrt(epsilon * mpi**2 * fpi**2 / (4*fa**2)) * np.sqrt(4*mu*md / (mu + md)**2)

        myvals = np.array([nNcentral, epsilon, fa, ma, r0, delta1, delta1cov])
        print(myvals)
        print("done with file " + str(index))

        total_data[index-indices[0], :] = myvals
        print(np.shape(total_data))
        np.save("/u/wentzel4/axions-in-NS/total-dynamics/outputs/domain-wall-analysis.npy", total_data)

    except Exception as e:
        print(e)
        print("error with file " + str(index))



