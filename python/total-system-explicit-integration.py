########################## INFORMATION ######################
# this file contains the complete integration for the axion, #
# baryon, and metric dynamics using a RKF4(5) integrator.    #
# 
#
# EXAMPLE INPUT LINE:
#      python3 total-system-explicit-integration.py "/Users/wentmich/Documents/uiuc/research/axions-in-NS/python/" 1000 16 25 0 0 5000 1600 10000 20 RK4
#
# this will run the command where all the outputs are sent to the directory above.
# epsilon is 1/1000
# fa is 10^16
# it's running using the density profile with the 25th  central density
# there's no intermediate metric integrator
# the initial time is 0 m
# the final time is 5000 m
# there are 1600 time steps
# there are 10000 radial steps
# the output is recorded every 20 time steps
# the integrator is RK4
#
#######################################################################################################################################

# get inputs from terminal
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import time as time
import scipy.ndimage as scimage
import math
from sympy.parsing.mathematica import mathematica
from numpy import cos, sin, tan, roll
from sys import argv

directory = argv[1]
epsilon   = 1/float(argv[2])
fa        = 10**float(argv[3])
myindex   = int(argv[4])
INTERMEDIATEMETRICINTEGRATOR = argv[5]
tinit, tfin = float(argv[6]), float(argv[7])
Nt, Nr = int(argv[8]), int(argv[9])
RESOLUTION = int(argv[10])
INTEGRATOR = argv[11]

dt = (tfin - tinit) / Nt

tvals = np.linspace(tinit, tfin, Nt)

EOSfilelocation = "/Users/wentmich/Documents/uiuc/research/axions-in-NS/EOS-MR/input_stable_eos_files_p_of_nb_fixed/"

# plotting instructions
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['text.usetex'] = True



# definition of some trig functions and absolute value
def sec(var):
    return 1.0 / cos(var);

def cot(var):
    return 1.0 / tan(var);

absapprox = 0.0
def myabs(cosa):
    return pow(cosa**2 + absapprox, 0.5);

potoffset = np.sqrt(1.0 + absapprox)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[0:box_pts] = y[0:box_pts]
    y_smooth[-2] = y[-2]
    y_smooth[-1] = y[-1]
    return y_smooth;



# constant definitions
# CONSTANTS IN GeV
Msun      = 1.989*10**30
GN        = 6.67*10**(-11)
G         = 1
clight    = 2.99792458*10**8
eV2joules = 1.60218*10**(-19);
GeV2joules= 1.60218*10**(-10);
mN        = 0.939563
hbarGeV   = 6.582119569*10**(-25)
hbarJ     = 1.05457*10**(-34)
hbarGeo   = hbarJ * (GN / clight**4) * clight
sigmaN   = 0.059
rkm       = 10**3 / (hbarGeV*clight)
rm        = 1/(hbarGeV*clight)
mpi       = 0.134977
fpi       = 0.130;
mu        = 0.0017;
md        = 0.0041;
RSTART    = 10**(-8);
RHOFRAC   = 10**(-12);
Ntot      = 95
E         = np.e
Pi        = np.pi

MeVperfm3_2_Jperm3 = (10**(-3) * GeV2joules) * (10**(-15))**(-3);
Jperm3_2_m2        = GN / clight**4;
GeV2m              = GeV2joules * GN / clight**4;

myscales  = np.array([10**4, 10**(-10), 1])
rscale = myscales[0]

ALIMITCUT = 1.0e-10
CUTOFFDIVIDEBYZERO = 1.2360600851388018e-16


# import central densities in case you need them
central_densities = np.loadtxt(directory + "EOS-1-functions/" + "central-density-vals.csv", delimiter=",")



# import and define equation of state and derivative
# IMPORT THE EOS FUNCTION 
EOSdata = np.loadtxt(EOSfilelocation + "EOS-1-m2.csv", dtype=float, delimiter=",")
density_low_domain = EOSdata[0, 0]
domainvals = np.append(EOSdata[:, 0], 1.0e-6)
rangevals  = np.append(EOSdata[:, 1], 9.350925465e-7)
drhovals = 12.0 * (np.roll(domainvals, -1, 0) - domainvals)
dpvals = -np.roll(rangevals, -2, 0) + 8.0 * np.roll(rangevals, -1, 0) - 8.0 * np.roll(rangevals, 1, 0) + np.roll(rangevals, 2, 0)
dpdrhovals = dpvals / drhovals

EOS_int = interpol.PchipInterpolator(domainvals, rangevals)
dEOSdata = np.loadtxt(EOSfilelocation + "dEOSdrho-1-m2.csv", dtype=float, delimiter=",")
dEOSdrho_int = interpol.PchipInterpolator(dEOSdata[:, 0], dEOSdata[:, 1])

EOS_slope = (rangevals[1] - rangevals[0]) / (domainvals[1] - domainvals[0])

def EOS(rho):
    return np.piecewise(rho, [rho > density_low_domain, rho <= density_low_domain], [lambda var: EOS_int(var), 0.0])

def dEOSdrho(rho):
    return np.piecewise(rho, [rho > density_low_domain, rho <= density_low_domain], [lambda var: dEOSdrho_int(var), 0.0])



# import density vals and define an interpolating function
density_vals = np.loadtxt(directory + "EOS-1-functions/" + "density-vals-rhoc-" + str(myindex) + ".csv", delimiter=",")
densityfnc = interpol.PchipInterpolator(density_vals[:, 0], density_vals[:, 1])

rinit, rfin = density_vals[0, 0], density_vals[-1, 0]
rvals = np.linspace(rinit, rfin, Nr)
dr = (rfin - rinit) / Nr

i = 0
while density_vals[i,1] > 0.0:
    i += 1

lengthofdensityvals = np.shape(density_vals)[0]
radiusNSindex = int(Nr * i / lengthofdensityvals)

# define first and second finite difference derivatives
# 8th order centered finite difference coefficients
cm4 = 1/280
cm3 = -4/105
cm2 = 1/5
cm1 = -4/5
c0  = 0 
c1  = 4/5
c2  = -1/5
c3  = 4/105
c4  = -1/280

def first_r_derivative_central_8(farray):
    first_deriv = cm4*np.roll(farray, 4, 0) + cm3*np.roll(farray, 3, 0) + cm2*np.roll(farray, 2, 0) + cm1*np.roll(farray, 1, 0) + c4*np.roll(farray, -4, 0) + c3*np.roll(farray, -3, 0) + c2*np.roll(farray, -2, 0) + c1*np.roll(farray, -1, 0)
    return first_deriv / dr;


# 8th order centered finite difference coefficients 2nd derivative
dm4 = -1/560
dm3 = 8/315
dm2 = -1/5
dm1 = 8/5
d0  = -205/72
d1  = 8/5
d2  = -1/5
d3  = 8/315
d4  = -1/560

def second_r_derivative_central_8(farray):
    second_deriv = dm4*np.roll(farray, 4, 0) + dm3*np.roll(farray, 3, 0) + dm2*np.roll(farray, 2, 0) + dm1*np.roll(farray, 1, 0) + d0*farray + d4*np.roll(farray, -4, 0) + d3*np.roll(farray, -3, 0) + d2*np.roll(farray, -2, 0) + d1*np.roll(farray, -1, 0)
    return second_deriv / dr**2;

# 6th order forward finite difference coefficients
f0 = -49/20
f1 = 6
f2 = -15/2
f3 = 20/3
f4  = -15/4
f5  = 6/5
f6  = -1/6

def first_r_derivative_forward_6(farray):
    first_deriv = f6*np.roll(farray, -6, 0) + f5*np.roll(farray, -5, 0) + f4*np.roll(farray, -4, 0) + f3*np.roll(farray, -3, 0) + f2*np.roll(farray, -2, 0) + f1*np.roll(farray, -1, 0) + f0*farray
    return first_deriv / dr;

def first_r_derivative_backward_6(farray):
    first_deriv = f6*np.roll(farray, 6, 0) + f5*np.roll(farray, 5, 0) + f4*np.roll(farray, 4, 0) + f3*np.roll(farray, 3, 0) + f2*np.roll(farray, 2, 0) + f1*np.roll(farray, 1, 0) + f0*farray
    return -first_deriv / dr;


# 6th order forward finite difference coefficients 2nd derivative
g0 = 469/90
g1 = -223/10
g2 = 879/20
g3 = -949/18
g4  = 41
g5  = -201/10
g6  = 1019/180
g7 = -7/10

def second_r_derivative_forward_6(farray):
    second_deriv = g7*np.roll(farray, -7, 0) + g6*np.roll(farray, -6, 0) + g5*np.roll(farray, -5, 0) + g4*np.roll(farray, -4, 0) + g3*np.roll(farray, -3, 0) + g2*np.roll(farray, -2, 0) + g1*np.roll(farray, -1, 0) + g0*farray
    return second_deriv / dr**2;

def second_r_derivative_backward_6(farray):
    second_deriv = g7*np.roll(farray, 7, 0) + g6*np.roll(farray, 6, 0) + g5*np.roll(farray, 5, 0) + g4*np.roll(farray, 4, 0) + g3*np.roll(farray, 3, 0) + g2*np.roll(farray, 2, 0) + g1*np.roll(farray, 1, 0) + g0*farray
    return second_deriv / dr**2;



def first_r_derivative(farray):
    first_deriv = np.zeros(np.shape(farray))
    first_deriv[6:-6] = first_r_derivative_central_8(farray)[6:-6]
    first_deriv[0:6]  = first_r_derivative_forward_6(farray)[0:6]
    first_deriv[-6:]= first_r_derivative_backward_6(farray)[-6:]
    first_deriv[0] = 0.0
    return first_deriv;

def second_r_derivative(farray):
    second_deriv = np.zeros(np.shape(farray))
    second_deriv[6:-6] = second_r_derivative_central_8(farray)[6:-6]
    second_deriv[0:6]  = second_r_derivative_forward_6(farray)[0:6]
    second_deriv[-6:]= second_r_derivative_backward_6(farray)[-6:]
    return second_deriv;




# define a piecewise-division-by-zero
def piecewise_division_by_zero(a, b):
    result = np.zeros(np.shape(a))
    np.divide(a, b, out=result, where=(abs(b) >= CUTOFFDIVIDEBYZERO))
    return result;



# define the TOV density profile
# density profile takes in r in meters and spits out density in m^-2
def density_profile(r):
    return np.piecewise(r, [r < density_vals[-1, 0], r >= density_vals[-1, 0]], [lambda var: densityfnc(var), lambda var: densityfnc(density_vals[-1, 0])*np.exp(-var + density_vals[-1, 0])])



####################### Runge-Kutta for Spatial Integration #######################
def inside_density(r, G0, U0, R0, A0, P0, dAdr):
    return (-1 + myabs(cos(A0/2.)))*pow(hbarGeo,-3)*pow(mN,-1)*(R0*sigmaN*pow(hbarGeo,3) - epsilon*mN*pow(fpi,2)*pow(GeV2m,4)*pow(mpi,2)) + (pow(G0,-2)*pow(GeV2m,2)*pow(hbarGeo,-1)*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2)))/2. + (R0 + EOS(R0)*pow(U0,2))*pow(1 - pow(U0,2),-1)

def get_total_inside_mass(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray):
    integral = 0
    for i in range(len(rvals)-1):
        integral += (dr/2) * 4 * np.pi * (rvals[i]**2 * inside_density(rvals[i], G0array[i], U0array[i], R0array[i], A0array[i], P0array[i], dAdrarray[i]) + rvals[i+1]**2 * inside_density(rvals[i+1], G0array[i+1], U0array[i+1], R0array[i+1], A0array[i+1], P0array[i+1], dAdrarray[i+1]))
    return integral;

def get_total_inside_mass_fast(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray):
    integral = (dr/2) * 4 * np.pi * (rvals**2 * inside_density(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray) + np.roll(rvals, -1, 0)**2 * inside_density(np.roll(rvals, -1, 0), np.roll(G0array, -1, 0), np.roll(U0array, -1, 0), np.roll(R0array, -1, 0), np.roll(A0array, -1, 0), np.roll(P0array, -1, 0), np.roll(dAdrarray, -1, 0)))
    integral = np.sum(integral) - integral[-1]
    return integral;


def k1Gr_G1(r, U0, R0, A0, P0, dAdr):
    return 4.212591890594653e-38*r*pow(dAdr,2)*pow(fa,2) + 4.212591890594653e-38*r*pow(fa,2)*pow(P0,2) + pow(r,-1)/2.

def k1Gr_G3(r, U0, R0, A0, P0, dAdr):
    return 6.662174619034264e-10*epsilon*r - 0.789107134111487*r*R0 + (-6.662174619034264e-10*epsilon*r + 0.789107134111487*r*R0)*myabs(cos(A0/2.)) - pow(r,-1)/2. - 4*Pi*r*R0*pow(-1 + pow(U0,2),-1) - 4*Pi*r*EOS(R0)*pow(U0,2)*pow(-1 + pow(U0,2),-1)

def k1Gr_G5(r, U0, R0, A0, P0, dAdr):
    return 0.0*rvals;

def k1Fr_F1(r, G0, U0, R0, A0, P0, dAdr):
    return pow(r,-1)*(0.5 - 4.212591890594653e-38*pow(fa,2)*pow(P0,2)*pow(r,2) + pow(dAdr,2)*pow(fa,2)*pow(r,2)*(-4.212591890594653e-38 + 4.212591890594653e-38*pow(U0,2)) - 0.5*pow(U0,2) + 4.212591890594653e-38*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + pow(G0,2)*(-0.5 + 6.662174619034264e-10*epsilon*pow(r,2) - 0.789107134111487*R0*pow(r,2) - 12.566370614359172*EOS(R0)*pow(r,2) + 0.5*pow(U0,2) - 6.662174619034264e-10*epsilon*pow(r,2)*pow(U0,2) - 11.777263480247685*R0*pow(r,2)*pow(U0,2) + myabs(cos(A0/2.))*pow(r,2)*(-6.662174619034264e-10*epsilon + 0.789107134111487*R0 + (6.662174619034264e-10*epsilon - 0.789107134111487*R0)*pow(U0,2))))*pow(-1. + pow(U0,2),-1)

def solve_for_G_next_step(r, G0, U0, R0, A0, P0, dAdr, dRdr):
    k1Gr = k1Gr_fnc(r, G0, U0, R0, A0, P0, dAdr)
    k2Gr = k1Gr_fnc(r + dr/2, G0 + k1Gr*dr/2, U0, R0, A0, P0, dAdr)
    k3Gr = k1Gr_fnc(r + dr/2, G0 + k2Gr*dr/2, U0, R0, A0, P0, dAdr)
    k4Gr = k1Gr_fnc(r + dr, G0 + k3Gr*dr, U0, R0, A0, P0, dAdr)
    return G0 + (k1Gr/6 + k2Gr/3 + k3Gr/3 + k4Gr/6) * dr;

def solve_for_G_next_step_fast(r, G0, G1c1, G3c1, G5c1, G1c2, G3c2, G5c2, G1c3, G3c3, G5c3, G1c4, G3c4, G5c4):
    k1Gr = G1c1 * G0 + G3c1 * G0**3 + G5c1 * G0**5
    k2Gr = G1c2 * (G0 + k1Gr*dr/2) + G3c2 * (G0 + k1Gr*dr/2)**3 + G5c2 * (G0 + k1Gr*dr/2)**5
    k3Gr = G1c3 * (G0 + k2Gr*dr/2) + G3c3 * (G0 + k2Gr*dr/2)**3 + G5c3 * (G0 + k2Gr*dr/2)**5
    k4Gr = G1c4 * (G0 + k3Gr*dr) + G3c4 * (G0 + k3Gr*dr)**3 + G5c4 * (G0 + k3Gr*dr)**5
    return G0 + (k1Gr/6 + k2Gr/3 + k3Gr/3 + k4Gr/6) * dr;

def solve_for_G_and_U_next_step(r, G0, U0, R0, A0, P0, dAdr, dRdr):
    k1Gr = k1Gr_fnc(r, G0, U0, R0, A0, P0, dAdr)
    k1Ur = k1Ur_fnc(r, G0, U0, R0, A0, P0, dAdr, dRdr)
    # get k2 arrays
    k2Gr = k1Gr_fnc(r + dr/2, G0 + k1Gr*dr/2, U0 + k1Ur*dr/2, R0, A0, P0, dAdr)
    k2Ur = k1Ur_fnc(r + dr/2, G0 + k1Gr*dr/2, U0 + k1Ur*dr/2, R0, A0, P0, dAdr, dRdr)
    # get k3 arrays
    k3Gr = k1Gr_fnc(r + dr/2, G0 + k2Gr*dr/2, U0 + k2Ur*dr/2, R0, A0, P0, dAdr)
    k3Ur = k1Ur_fnc(r + dr/2, G0 + k2Gr*dr/2, U0 + k2Ur*dr/2, R0, A0, P0, dAdr, dRdr)
    # get k4 arrays
    k4Gr = k1Gr_fnc(r + dr, G0 + k3Gr*dr, U0 + k3Ur*dr, R0, A0, P0, dAdr)
    k4Ur = k1Ur_fnc(r + dr, G0 + k3Gr*dr, U0 + k3Ur*dr, R0, A0, P0, dAdr, dRdr)
    # update with Runge-Kutta
    G1 = G0 + (1/6)*k1Gr*dr + (1/3)*k1Gr*dr + (1/3)*k1Gr*dr + (1/6)*k1Gr*dr
    U1 = U0 + (1/6)*k1Ur*dr + (1/3)*k1Ur*dr + (1/3)*k1Ur*dr + (1/6)*k1Ur*dr
    return np.array([G1, U1]).astype(np.float64);
    

def solve_for_F_next_step(r, G0, F0, U0, R0, A0, P0, dAdr):
    k1Fr = k1Fr_fnc(r, G0, F0, U0, R0, A0, P0, dAdr)
    k2Fr = k1Fr_fnc(r + dr/2, G0, F0 + k1Fr*dr/2, U0, R0, A0, P0, dAdr)
    k3Fr = k1Fr_fnc(r + dr/2, G0, F0 + k2Fr*dr/2, U0, R0, A0, P0, dAdr)
    k4Fr = k1Fr_fnc(r + dr, G0, F0 + k3Fr*dr, U0, R0, A0, P0, dAdr)
    return F0 - (k1Fr/6 + k2Fr/3 + k3Fr/3 + k4Fr/6) * dr;

def solve_for_F_next_step_fast(r, F0, F1c1, F1c2, F1c3, F1c4):
    k1Fr = F1c1 * F0
    k2Fr = F1c2 * (F0 - k1Fr*dr/2) 
    k3Fr = F1c3 * (F0 - k2Fr*dr/2)
    k4Fr = F1c4 * (F0 - k3Fr*dr)
    return F0 - (k1Fr/6 + k2Fr/3 + k3Fr/3 + k4Fr/6) * dr;


def solve_for_G(G0point, U0array, R0array, A0array, P0array, dAdrarray, dRdrarray):
    G1vals = np.zeros(np.shape(rvals))
    G1vals[0] = G0point
    for i in range(1, Nr):
        G1vals[i] = solve_for_G_next_step(rvals[i-1], G1vals[i-1], U0array[i-1], R0array[i-1], A0array[i-1], P0array[i-1], dAdrarray[i-1], dRdrarray[i-1])
    
    return G1vals;

def solve_for_G_fast(G0point, U0array, R0array, A0array, P0array, dAdrarray, dRdrarray):
    U0array2 = (np.roll(U0array, -1, 0) + U0array)/2.0
    U0array2[-1] = U0array[-1]
    R0array2 = (np.roll(R0array, -1, 0) + R0array)/2.0
    R0array2[-1] = R0array[-1]
    A0array2 = (np.roll(A0array, -1, 0) + A0array)/2.0
    A0array2[-1] = A0array[-1]
    P0array2 = (np.roll(P0array, -1, 0) + P0array)/2.0
    P0array2[-1] = P0array[-1]
    dAdrarray2 = (np.roll(dAdrarray, -1, 0) + dAdrarray)/2.0
    dAdrarray2[-1] = dAdrarray[-1]
    
    U0array4 = np.roll(U0array, -1, 0)
    U0array4[-1] = U0array[-1]
    R0array4 = np.roll(R0array, -1, 0)
    R0array4[-1] = R0array[-1]
    A0array4 = np.roll(A0array, -1, 0)
    A0array4[-1] = A0array[-1]
    P0array4 = np.roll(P0array, -1, 0)
    P0array4[-1] = P0array[-1]
    dAdrarray4 = np.roll(dAdrarray, -1, 0)
    dAdrarray4[-1] = dAdrarray[-1]
    
    G1c1 = k1Gr_G1(rvals, U0array, R0array, A0array, P0array, dAdrarray)
    G3c1 = k1Gr_G3(rvals, U0array, R0array, A0array, P0array, dAdrarray)
    G5c1 = k1Gr_G5(rvals, U0array, R0array, A0array, P0array, dAdrarray)
    
    G1c2 = k1Gr_G1(rvals + np.ones(np.shape(rvals))*dr/2, U0array2, R0array2, A0array2, P0array2, dAdrarray2)
    G3c2 = k1Gr_G3(rvals + np.ones(np.shape(rvals))*dr/2, U0array2, R0array2, A0array2, P0array2, dAdrarray2)
    G5c2 = k1Gr_G5(rvals + np.ones(np.shape(rvals))*dr/2, U0array2, R0array2, A0array2, P0array2, dAdrarray2)
    
    G1c4 = k1Gr_G1(rvals + np.ones(np.shape(rvals))*dr, U0array4, R0array4, A0array4, P0array4, dAdrarray4)
    G3c4 = k1Gr_G3(rvals + np.ones(np.shape(rvals))*dr, U0array4, R0array4, A0array4, P0array4, dAdrarray4)
    G5c4 = k1Gr_G5(rvals + np.ones(np.shape(rvals))*dr, U0array4, R0array4, A0array4, P0array4, dAdrarray4)
    
    G1vals = np.zeros(np.shape(rvals))
    G1vals[0] = G0point
    for i in range(1, Nr):
        G1vals[i] = solve_for_G_next_step_fast(rvals[i-1], G1vals[i-1], G1c1[i-1], G3c1[i-1], G5c1[i-1], G1c2[i-1], G3c2[i-1], G5c2[i-1], G1c2[i-1], G3c2[i-1], G5c2[i-1], G1c4[i-1], G3c4[i-1], G5c4[i-1])
    
    return G1vals;

def solve_for_G_and_U(G0point, U0point, R0array, A0array, P0array, dAdrarray, dRdrarray):
    G1vals = np.zeros(np.shape(rvals))
    U1vals = np.zeros(np.shape(rvals))
    G1vals[0] = G0point
    U1vals[0] = U0point
    for i in range(1, Nr):
        newvals = solve_for_G_and_U_next_step(rvals[i-1], G1vals[i-1], U1vals[i-1], R0array[i-1], A0array[i-1], P0array[i-1], dAdrarray[i-1], dRdrarray[i-1])
        G1vals[i] = newvals[0]
        U1vals[i] = newvals[1]
        
    return np.concatenate((G1vals, U1vals)).astype(np.float64);

def solve_for_F(G0array, U0array, R0array, A0array, P0array, dAdrarray):
    F1vals = np.zeros(np.shape(rvals))
    F1vals[-1] = np.sqrt(1 - 2*get_total_inside_mass(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray) / rvals[-1])
    for i in range(2, Nr+1):
        F1vals[-i] = solve_for_F_next_step(rvals[-i+1], G0array[-i+1], F1vals[-i+1], U0array[-i+1], R0array[-i+1], A0array[-i+1], P0array[-i+1], dAdrarray[-i+1])
    
    return F1vals;

def solve_for_F_fast(G0array, U0array, R0array, A0array, P0array, dAdrarray):
    G0array2 = (np.roll(G0array, 1, 0) + G0array) / 2.0
    G0array2[0] = G0array[0]
    U0array2 = (np.roll(U0array, 1, 0) + U0array) / 2.0
    U0array2[0] = U0array[0]
    R0array2 = (np.roll(R0array, 1, 0) + R0array) / 2.0
    R0array2[0] = R0array[0]
    A0array2 = (np.roll(A0array, 1, 0) + A0array) / 2.0
    A0array2[0] = A0array[0]
    P0array2 = (np.roll(P0array, 1, 0) + P0array) / 2.0
    P0array2[0] = P0array[0]
    dAdrarray2 = (np.roll(dAdrarray, 1, 0) + dAdrarray) / 2.0
    dAdrarray2[0] = dAdrarray[0]
    
    G0array4 = np.roll(G0array, 1, 0)
    G0array4[0] = G0array[0]
    U0array4 = np.roll(U0array, 1, 0)
    U0array4[0] = U0array[0]
    R0array4 = np.roll(R0array, 1, 0)
    R0array4[0] = R0array[0]
    A0array4 = np.roll(A0array, 1, 0)
    A0array4[0] = A0array[0]
    P0array4 = np.roll(P0array, 1, 0)
    P0array4[0] = P0array[0]
    dAdrarray4 = np.roll(dAdrarray, 1, 0)
    dAdrarray4[0] = dAdrarray[0]
    
    F1c1 = k1Fr_F1(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray)
    
    F1c2 = k1Fr_F1(rvals - np.ones(np.shape(rvals))*dr/2, G0array2, U0array2, R0array2, A0array2, P0array2, dAdrarray2)
    
    F1c4 = k1Fr_F1(rvals - np.ones(np.shape(rvals))*dr, G0array4, U0array4, R0array4, A0array4, P0array4, dAdrarray4)
    
    F1vals = np.zeros(np.shape(rvals))
    integraltime = time.time()
    F1vals[-1] = np.sqrt(1 - 2*get_total_inside_mass_fast(rvals, G0array, U0array, R0array, A0array, P0array, dAdrarray) / rvals[-1])
    integraltotaltime = time.time() - integraltime
    
    for i in range(2, Nr+1):
        F1vals[-i] = solve_for_F_next_step_fast(rvals[-i+1], F1vals[-i+1], F1c1[-i+1], F1c2[-i+1], F1c2[-i+1], F1c4[-i+1])
    
    return F1vals;

def metric_integrator(U0array, R0array, A0array, P0array):
    dAdrarray = first_r_derivative(A0array)
    dRdrarray = first_r_derivative(R0array)
    G1vals = solve_for_G_fast(1, U0array, R0array, A0array, P0array, dAdrarray, dRdrarray)
    F1vals = solve_for_F_fast(G1vals, U0array, R0array, A0array, P0array, dAdrarray)
    return np.concatenate((F1vals, G1vals)).astype(np.float64);




###################### Runge-Kutta Integrator ########################
# the runge-kutta integrator will take initial conditions for the following six functions:
# U - radial 4 velocity scaled by G, U = G*ur
# R - baryon density
# A - axion field scaled by fa, A = a*fa
# P - axion velocity scaled by metric components, P = (G/F)*ak1
# it then steps them up using a 4th order Runge-Kutta
# for the spatial derivatives it uses a 4th order finite difference

def zero_below(myarrary, epsilon):
    result = myarrary.copy()
    result[result < epsilon] = 0.0
    return result;


def custom_sign(myarray):
    return np.where(myarray == 0, -1, np.sign(myarray));


def kiG_function(r, F0, G0, U0, R0, A0, P0):
    dAdr  = first_r_derivative(A0)
    k1G = 4*F0*Pi*r*(dAdr*P0*pow(fa,2)*pow(GeV2m,2)*pow(hbarGeo,-1) + U0*(R0 + EOS(R0))*pow(G0,2)*pow(-1 + pow(U0,2),-1))
    return k1G.astype(np.float64);


def a20_where_bis0(a, b):
    result = np.zeros(np.shape(a))
    np.divide(a*b, b, out=result, where=(b > 0.0))
    return result


def kiU_function(r, F0, G0, U0, R0, A0, P0):
    dUdr  = first_r_derivative(U0)
    dAdr  = first_r_derivative(A0)
    dRdr  = first_r_derivative(R0)
    
    numerator = F0*pow(G0,-1)*pow(r,-1)*(pow(G0,2)*pow(r,2)*(-187.55052554732126 + 187.55052554732126*pow(U0,2) + myabs(cos(A0/2.))*(-12.566370614359172 + 12.566370614359172*pow(U0,2)))*pow(EOS(R0),2) + pow(G0,2)*pow(r,2)*pow(R0,2)*(-11.777263480247685 - 187.55052554732134*dEOSdrho(R0)*pow(U0,2) + (11.777263480247685 + 187.55052554732134*dEOSdrho(R0))*pow(U0,4) + myabs(cos(A0/2.))*(10.9881563461362 - 12.566370614359174*dEOSdrho(R0)*pow(U0,2) + (-10.9881563461362 + 12.566370614359174*dEOSdrho(R0))*pow(U0,4)) + (0.7891071341114871 - 0.7891071341114871*pow(U0,4))*pow(myabs(cos(A0/2.)),2)) + dRdr*r*(-0.9372048494885389 - 14.92479661016949*dEOSdrho(R0) + (1.8744096989770778 + 29.84959322033898*dEOSdrho(R0))*pow(U0,2) + (-0.9372048494885389 - 14.92479661016949*dEOSdrho(R0))*pow(U0,4) + myabs(cos(A0/2.))*(0.8744096989770775 - 1.*dEOSdrho(R0) + (-1.748819397954155 + 2.*dEOSdrho(R0))*pow(U0,2) + (0.8744096989770775 - 1.*dEOSdrho(R0))*pow(U0,4)) + (0.06279515051146116 - 0.12559030102292232*pow(U0,2) + 0.06279515051146116*pow(U0,4))*pow(myabs(cos(A0/2.)),2)) + R0*(7.462398305084745 - 13.924796610169492*dUdr*r*U0 + 15.92479661016949*dUdr*r*U0*dEOSdrho(R0) - 1.1731897055635982e-36*dAdr*P0*U0*pow(fa,2)*pow(r,2) + 1.3416933811873843e-36*dAdr*P0*U0*dEOSdrho(R0)*pow(fa,2)*pow(r,2) - 6.287207716877456e-37*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 6.287207716877456e-37*pow(fa,2)*pow(P0,2)*pow(r,2) - 5.962398305084745*pow(U0,2) + 23.88719491525424*dEOSdrho(R0)*pow(U0,2) + 6.708466905936922e-37*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*dEOSdrho(R0)*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*dEOSdrho(R0)*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + 1.1731897055635982e-36*dAdr*P0*pow(fa,2)*pow(r,2)*pow(U0,3) - 1.3416933811873843e-36*dAdr*P0*dEOSdrho(R0)*pow(fa,2)*pow(r,2)*pow(U0,3) - 1.5*pow(U0,4) - 23.88719491525424*dEOSdrho(R0)*pow(U0,4) - 4.2125918905946527e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,4) - 6.708466905936922e-37*dEOSdrho(R0)*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,4) - 4.2125918905946527e-38*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,4) - 6.708466905936922e-37*dEOSdrho(R0)*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,4) + myabs(cos(A0/2.))*(0.5 + r*U0*(-2.*dUdr - 1.685036756237861e-37*dAdr*P0*r*pow(fa,2)) - 4.2125918905946527e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 4.2125918905946527e-38*pow(fa,2)*pow(P0,2)*pow(r,2) - 2.*pow(U0,2) + 1.685036756237861e-37*dAdr*P0*pow(fa,2)*pow(r,2)*pow(U0,3) + (1.5 + 4.2125918905946527e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 4.2125918905946527e-38*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,4)) + pow(G0,2)*(-7.462398305084745 + 9.94316011705198e-9*epsilon*pow(r,2) + (7.962398305084746 - 1.0609377578955407e-8*epsilon*pow(r,2) + dEOSdrho(R0)*(7.962398305084746 - 1.0609377578955407e-8*epsilon*pow(r,2)))*pow(U0,2) + (-0.5 + 6.662174619034264e-10*epsilon*pow(r,2) + dEOSdrho(R0)*(-7.962398305084746 + 1.0609377578955407e-8*epsilon*pow(r,2)))*pow(U0,4) + myabs(cos(A0/2.))*(-0.5 - 9.276942655148554e-9*epsilon*pow(r,2) + epsilon*(1.0609377578955407e-8 + 1.0609377578955407e-8*dEOSdrho(R0))*pow(r,2)*pow(U0,2) + (0.5 - 1.3324349238068528e-9*epsilon*pow(r,2) - 1.0609377578955407e-8*epsilon*dEOSdrho(R0)*pow(r,2))*pow(U0,4)) + epsilon*pow(r,2)*(-6.662174619034264e-10 + 6.662174619034264e-10*pow(U0,4))*pow(myabs(cos(A0/2.)),2))) + EOS(R0)*(7.462398305084745 - 13.924796610169492*dUdr*r*U0 + 15.92479661016949*dUdr*r*U0*dEOSdrho(R0) - 1.1731897055635982e-36*dAdr*P0*U0*pow(fa,2)*pow(r,2) + 1.3416933811873843e-36*dAdr*P0*U0*dEOSdrho(R0)*pow(fa,2)*pow(r,2) - 6.287207716877456e-37*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 6.287207716877456e-37*pow(fa,2)*pow(P0,2)*pow(r,2) - 5.962398305084744*pow(U0,2) + 23.88719491525424*dEOSdrho(R0)*pow(U0,2) + 6.708466905936922e-37*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*dEOSdrho(R0)*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + 6.708466905936922e-37*dEOSdrho(R0)*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + 1.1731897055635982e-36*dAdr*P0*pow(fa,2)*pow(r,2)*pow(U0,3) - 1.3416933811873843e-36*dAdr*P0*dEOSdrho(R0)*pow(fa,2)*pow(r,2)*pow(U0,3) - 1.4999999999999996*pow(U0,4) - 23.88719491525424*dEOSdrho(R0)*pow(U0,4) - 4.2125918905946517e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,4) - 6.708466905936922e-37*dEOSdrho(R0)*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,4) - 4.2125918905946517e-38*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,4) - 6.708466905936922e-37*dEOSdrho(R0)*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,4) + myabs(cos(A0/2.))*(0.5 + r*U0*(-2.*dUdr - 1.6850367562378607e-37*dAdr*P0*r*pow(fa,2)) - 4.2125918905946517e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 4.2125918905946517e-38*pow(fa,2)*pow(P0,2)*pow(r,2) - 2.*pow(U0,2) + 1.6850367562378607e-37*dAdr*P0*pow(fa,2)*pow(r,2)*pow(U0,3) + (1.4999999999999996 + 4.2125918905946517e-38*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 4.2125918905946517e-38*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,4)) + pow(G0,2)*(-7.462398305084745 + 9.94316011705198e-9*epsilon*pow(r,2) - 199.32778902756897*R0*pow(r,2) + (7.962398305084745 - 1.0609377578955407e-8*epsilon*pow(r,2) + R0*(187.55052554732126 - 187.55052554732134*dEOSdrho(R0))*pow(r,2) + dEOSdrho(R0)*(7.962398305084746 - 1.0609377578955407e-8*epsilon*pow(r,2)))*pow(U0,2) + (-0.5 + 6.662174619034264e-10*epsilon*pow(r,2) + R0*(11.777263480247685 + 187.55052554732134*dEOSdrho(R0))*pow(r,2) + dEOSdrho(R0)*(-7.962398305084746 + 1.0609377578955407e-8*epsilon*pow(r,2)))*pow(U0,4) + myabs(cos(A0/2.))*(-0.5 - 9.276942655148554e-9*epsilon*pow(r,2) - 1.5782142682229738*R0*pow(r,2) + (R0*(12.566370614359172 - 12.566370614359172*dEOSdrho(R0)) + epsilon*(1.0609377578955409e-8 + 1.0609377578955407e-8*dEOSdrho(R0)))*pow(r,2)*pow(U0,2) + (0.5 - 1.3324349238068526e-9*epsilon*pow(r,2) - 1.0609377578955407e-8*epsilon*dEOSdrho(R0)*pow(r,2) + R0*(-10.988156346136199 + 12.566370614359172*dEOSdrho(R0))*pow(r,2))*pow(U0,4)) + pow(r,2)*(-6.662174619034264e-10*epsilon + 0.7891071341114869*R0 + (6.662174619034264e-10*epsilon - 0.7891071341114869*R0)*pow(U0,4))*pow(myabs(cos(A0/2.)),2))))*pow(14.924796610169492 + (-1. - 15.924796610169492*dEOSdrho(R0))*pow(U0,2) + myabs(cos(A0/2.))*(1. + 1.*pow(U0,2)),-1)
    denominator = R0 + EOS(R0)
    
    finalkiU = np.zeros(np.shape(rvals)).astype(np.float64)
    np.divide(numerator.astype(np.float64), denominator.astype(np.float64), out=finalkiU, where=(denominator > 0.0))
    finalkiU[(radiusNSindex - 10):(radiusNSindex + 10)] = np.zeros(20)
    
    finalkiU[0:2] = np.zeros(2)
    
    return finalkiU.astype(np.float64)

    
def kiR_function(r, F0, G0, U0, R0, A0, P0):
    dUdr  = first_r_derivative(U0)
    dRdr  = first_r_derivative(R0)
    dAdr  = first_r_derivative(A0)
    dAdr2 = second_r_derivative(A0)
    
    k1Rarray = F0*pow(G0,-1)*pow(hbarGeo,-1)*pow(r,-1)*(mN*r*R0*(dUdr*hbarGeo + 4*dAdr*P0*Pi*r*pow(fa,2)*pow(GeV2m,2)) - hbarGeo*U0*(-2*mN*R0 + dRdr*r*(-mN + 2*sigmaN + mN*dEOSdrho(R0) - 2*sigmaN*myabs(cos(A0/2.))) + 4*mN*Pi*pow(G0,2)*pow(r,2)*pow(R0,2)) - 4*dAdr*mN*P0*Pi*R0*pow(fa,2)*pow(GeV2m,2)*pow(r,2)*pow(U0,2) + mN*EOS(R0)*(r*(dUdr*hbarGeo + 4*dAdr*P0*Pi*r*pow(fa,2)*pow(GeV2m,2)) + U0*(2*hbarGeo - 8*hbarGeo*Pi*R0*pow(G0,2)*pow(r,2)) - 4*dAdr*P0*Pi*pow(fa,2)*pow(GeV2m,2)*pow(r,2)*pow(U0,2)) - 4*hbarGeo*mN*Pi*U0*pow(G0,2)*pow(r,2)*pow(EOS(R0),2))*pow(-mN + sigmaN + (sigmaN + mN*dEOSdrho(R0))*pow(U0,2) - sigmaN*myabs(cos(A0/2.))*(1 + pow(U0,2)),-1)
    
    return k1Rarray.astype(np.float64);

def kiA_function(r, F0, G0, U0, R0, A0, P0):
    k1Aarray = (F0*fa*P0*pow(G0,-1)) / fa
    return k1Aarray.astype(np.float64);

def kiP_function_nonzero(r, F0, G0, U0, R0, A0, P0):
    dAdr  = first_r_derivative(A0)
    dAdr2 = second_r_derivative(A0)
    k1ParrayNonZero = (-4*dAdr*F0*fa*G0*Pi*r*R0 + 4*dAdr*F0*fa*G0*Pi*r*EOS(R0) + dAdr2*F0*fa*pow(G0,-1) + 8*dAdr*F0*fa*G0*Pi*r*R0*sigmaN*pow(mN,-1) - 8*dAdr*F0*fa*G0*Pi*r*R0*sigmaN*myabs(cos(A0/2.))*pow(mN,-1) - 8*dAdr*epsilon*F0*fa*G0*Pi*r*pow(fpi,2)*pow(GeV2m,4)*pow(hbarGeo,-3)*pow(mpi,2) + 8*dAdr*epsilon*F0*fa*G0*Pi*r*myabs(cos(A0/2.))*pow(fpi,2)*pow(GeV2m,4)*pow(hbarGeo,-3)*pow(mpi,2) + dAdr*F0*fa*G0*pow(r,-1) + dAdr*F0*fa*pow(G0,-1)*pow(r,-1)) / fa
    return k1ParrayNonZero.astype(np.float64);

def kiP_function_zero(r, F0, G0, U0, R0, A0, P0, alimitcut):
    k1ParrayCoefficient = (F0*G0*hbarGeo*R0*sigmaN*pow(fa,-1)*pow(GeV2m,-2)*pow(mN,-1))/4. - (epsilon*F0*G0*pow(fa,-1)*pow(fpi,2)*pow(GeV2m,2)*pow(hbarGeo,-2)*pow(mpi,2))/4.
    SinOverCos = (-2.0 * custom_sign(A0 - np.ones(np.shape(A0)) * np.pi))
    np.divide(sin(A0), myabs(cos(A0/2.0)), out=SinOverCos, where=(abs(A0 - np.ones(np.shape(A0)) * np.pi) >= alimitcut))
    k1ParrayZero = k1ParrayCoefficient * SinOverCos / fa
    
    return k1ParrayZero.astype(np.float64);

def kiP_function(r, F0, G0, U0, R0, A0, P0):
    k1Parray = kiP_function_nonzero(r, F0, G0, U0, R0, A0, P0) + kiP_function_zero(r, F0, G0, U0, R0, A0, P0, ALIMITCUT)
    
    return k1Parray.astype(np.float64);



def matter_integrator(r, F0, G0, U0, R0, A0, P0):
    # get k1 arrays
    k1U = kiU_function(r, F0, G0, U0, R0, A0, P0)
    k1U[0] = 0.0
    #print("k1U ", max(k1U))
    k1R = kiR_function(r, F0, G0, U0, R0, A0, P0)
    k1A = kiA_function(r, F0, G0, U0, R0, A0, P0)
    #print("k1A ", k1A)
    k1P = kiP_function(r, F0, G0, U0, R0, A0, P0)
    # get k2 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        metrichalf1 = metric_integrator(U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
        Fhalf1 = metrichalf1[0:Nr]
        Ghalf1 = metrichalf1[Nr:2*Nr]
    else:
        Fhalf1 = F0
        Ghalf1 = G0
    k2U = kiU_function(r, Fhalf1, Ghalf1, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2U[0] = 0.0
    k2R = kiR_function(r, Fhalf1, Ghalf1, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2A = kiA_function(r, Fhalf1, Ghalf1, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2P = kiP_function(r, Fhalf1, Ghalf1, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    # get k3 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        metrichalf2 = metric_integrator(U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
        Fhalf2 = metrichalf2[0:Nr]
        Ghalf2 = metrichalf2[Nr:2*Nr]
    else:
        Fhalf2 = F0
        Ghalf2 = G0
    k3U = kiU_function(r, Fhalf2, Ghalf2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3U[0] = 0.0
    k3R = kiR_function(r, Fhalf2, Ghalf2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3A = kiA_function(r, Fhalf2, Ghalf2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3P = kiP_function(r, Fhalf2, Ghalf2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    # get k4 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        metricfull = metric_integrator(U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
        Ffull = metricfull[0:Nr]
        Gfull = metricfull[Nr:2*Nr]
    else:
        Ffull = F0
        Gfull = G0
    k4U = kiU_function(r, Ffull, Gfull, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4U[0] = 0.0
    k4R = kiR_function(r, Ffull, Gfull, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4A = kiA_function(r, Ffull, Gfull, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4P = kiP_function(r, Ffull, Gfull, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    # update with Runge-Kutta
    U1 = U0 + (1/6)*k1U*dt + (1/3)*k2U*dt + (1/3)*k3U*dt + (1/6)*k4U*dt
    R1 = R0 + (1/6)*k1R*dt + (1/3)*k2R*dt + (1/3)*k3R*dt + (1/6)*k4R*dt
    A1 = A0 + (1/6)*k1A*dt + (1/3)*k2A*dt + (1/3)*k3A*dt + (1/6)*k4A*dt
    P1 = P0 + (1/6)*k1P*dt + (1/3)*k2P*dt + (1/3)*k3P*dt + (1/6)*k4P*dt
    
    return np.concatenate((U1, R1, A1, P1)).astype(np.float64);


def matter_and_G_integrator(r, F0, G0, U0, R0, A0, P0, indexflag):
    # get k1 arrays
    if indexflag == 1:
        k1U = np.zeros(Nr)
    else:
        k1U = kiU_function(r, F0, G0, U0, R0, A0, P0)
    k1U[0] = 0.0
    k1R = kiR_function(r, F0, G0, U0, R0, A0, P0)
    k1A = kiA_function(r, F0, G0, U0, R0, A0, P0)
    k1P = kiP_function(r, F0, G0, U0, R0, A0, P0)
    k1G = kiG_function(r, F0, G0, U0, R0, A0, P0)
    k1G[0] = 0.0
    #print("max of U1 with k1: ", max(abs(U0 + k1U*dt/2)))
    # get k2 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayhalf1 = first_r_derivative(A0 + k1A*dt/2)
        Fhalf1 = solve_for_F_fast(G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2, dAdrarrayhalf1)
    else:
        Fhalf1 = F0
    k2G = kiG_function(r, Fhalf1, G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2G[0] = 0.0
    if indexflag == 1:
        k2U = np.zeros(Nr)
    else:
        k2U = kiU_function(r, Fhalf1, G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2U[0] = 0.0
    k2R = kiR_function(r, Fhalf1, G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2A = kiA_function(r, Fhalf1, G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    k2P = kiP_function(r, Fhalf1, G0 + k1G*dt/2, U0 + k1U*dt/2, R0 + k1R*dt/2, A0 + k1A*dt/2, P0 + k1P*dt/2)
    #print("max of U1 with k2: ", max(abs(U0 + k2U*dt/2)))
    # get k3 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayhalf2 = first_r_derivative(A0 + k2A*dt/2)
        Fhalf2 = solve_for_F_fast(G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2, dAdrarrayhalf2)
    else:
        Fhalf2 = F0
    k3G = kiU_function(r, Fhalf2, G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3G[0] = 0.0
    if indexflag == 1:
        k3U = np.zeros(Nr)
    else:
        k3U = kiU_function(r, Fhalf2, G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3U[0] = 0.0
    k3R = kiR_function(r, Fhalf2, G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3A = kiA_function(r, Fhalf2, G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    k3P = kiP_function(r, Fhalf2, G0 + k2G*dt/2, U0 + k2U*dt/2, R0 + k2R*dt/2, A0 + k2A*dt/2, P0 + k2P*dt/2)
    #print("max of U1 with k3: ", max(abs(U0 + k3U*dt)))
    # get k4 arrays
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarrayfull = first_r_derivative(A0 + k3A*dt)
        Ffull = solve_for_F_fast(G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt, dAdrarrayfull)
    else:
        Ffull = F0
    k4G = kiU_function(r, Ffull, G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4G[0] = 0.0
    if indexflag == 1:
        k4U = np.zeros(Nr)
    else:
        k4U = kiU_function(r, Ffull, G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4U[0] = 0.0
    k4R = kiR_function(r, Ffull, G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4A = kiA_function(r, Ffull, G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    k4P = kiP_function(r, Ffull, G0 + k3G*dt, U0 + k3U*dt, R0 + k3R*dt, A0 + k3A*dt, P0 + k3P*dt)
    # update with Runge-Kutta
    G1 = G0 + (1/6)*k1G*dt + (1/3)*k2G*dt + (1/3)*k3G*dt + (1/6)*k4G*dt
    U1 = U0 + (1/6)*k1U*dt + (1/3)*k2U*dt + (1/3)*k3U*dt + (1/6)*k4U*dt
    R1 = R0 + (1/6)*k1R*dt + (1/3)*k2R*dt + (1/3)*k3R*dt + (1/6)*k4R*dt
    A1 = A0 + (1/6)*k1A*dt + (1/3)*k2A*dt + (1/3)*k3A*dt + (1/6)*k4A*dt
    P1 = P0 + (1/6)*k1P*dt + (1/3)*k2P*dt + (1/3)*k3P*dt + (1/6)*k4P*dt
    
    U1[0] = 0.0
    U1[1] = 0.0
    G1[0] = 1.0
    
    return np.concatenate((U1, R1, A1, P1, G1)).astype(np.float64);


# define the constants you need for RKF45
A1 = 0.
A2 = 2/9
A3 = 1/3
A4 = 3/4
A5 = 1.
A6 = 5.6

B21 = 2/9
B31 = 1/12
B32 = 1/4
B41 = 69/128
B42 = -243/128
B43 = 135/64
B51 = -17/12
B52 = 27/4
B53 = -27/5
B54 = 16/15
B61 = 65/432
B62 = -5/16
B63 = 13/16
B64 = 4/27
B65 = 5/144

C1 = 1/9
C2 = 0.
C3 = 9/20
C4 = 16/45
C5 = 1/12

CH1 = 47/450
CH2 = 0.
CH3 = 12/25
CH4 = 32/225
CH5 = 1/30
CH6 = 6/25

CT1 = 1/150
CT2 = 0.
CT3 = -3/100
CT4 = 16/75
CT5 = 1/20
CT6 = -6/25


truncation_error_strings = ["G", "U", "R", "A", "P"]


# now define the RHF45 integrator
def matter_and_G_integrator_RKF45(r, F0, G0, U0, R0, A0, P0, indexflag, tstep, acceptable_errors_fraction, verbose):
    acceptable_errors = acceptable_errors_fraction * np.array([1.0, 1.0, 10.0**(-10.0), np.pi, np.pi])
    # This should be approximately the same as the RK4 solver but with an extra
    # ki and an adaptive time step. That seems like we could run into issues
    # with the simulation time blowing up, but maybe not.
    # get k1 arrays
    k1U = tstep*kiU_function(r, F0, G0, U0, R0, A0, P0)
    #k1U[0] = 0.0
    k1R = tstep*kiR_function(r, F0, G0, U0, R0, A0, P0)
    k1A = tstep*kiA_function(r, F0, G0, U0, R0, A0, P0)
    k1P = tstep*kiP_function(r, F0, G0, U0, R0, A0, P0)
    k1G = tstep*kiG_function(r, F0, G0, U0, R0, A0, P0)
    #k1G[0] = 0.0
    
    # solve for the quarter step of the F2 metric function
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray2 = first_r_derivative(A0 + B21*k1A)
        Fint2 = solve_for_F_fast(G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P, dAdrarray2)
    else:
        Fint2 = F0
        
        
    k2G = tstep*kiG_function(r, Fint2, G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P)
    #k2G[0] = 0.0
    k2U = tstep*kiU_function(r, Fint2, G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P)
    #k2U[0] = 0.0
    k2R = tstep*kiR_function(r, Fint2, G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P)
    k2A = tstep*kiA_function(r, Fint2, G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P)
    k2P = tstep*kiP_function(r, Fint2, G0 + B21*k1G, U0 + B21*k1U, R0 + B21*k1R, A0 + B21*k1A, P0 + B21*k1P)

    # solve for F3 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray3 = first_r_derivative(A0 + B31*k1A + B32*k2A)
        Fint3 = solve_for_F_fast(G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P, dAdrarray3)
    else:
        Fint3 = F0
        
    
    k3G = tstep*kiU_function(r, Fint3, G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    #k3G[0] = 0.0
    k3U = tstep*kiU_function(r, Fint3, G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    #k3U[0] = 0.0
    k3R = tstep*kiR_function(r, Fint3, G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    k3A = tstep*kiA_function(r, Fint3, G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)
    k3P = tstep*kiP_function(r, Fint3, G0 + B31*k1G + B32*k2G, U0 + B31*k1U + B32*k2U, R0 + B31*k1R + B32*k2R, A0 + B31*k1A + B32*k2A, P0 + B31*k1P + B32*k2P)

    # solve for F4 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray4 = first_r_derivative(A0 + B41*k1A + B42*k2A + B43*k3A)
        Fint4 = solve_for_F_fast(G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P, dAdrarray4)
    else:
        Fint4 = F0
        
    
    k4G = tstep*kiU_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    #k4G[0] = 0.0
    k4U = tstep*kiU_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    #k4U[0] = 0.0
    k4R = tstep*kiR_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    k4A = tstep*kiA_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    k4P = tstep*kiP_function(r, Fint4, G0 + B41*k1G + B42*k2G + B43*k3G, U0 + B41*k1U + B42*k2U + B43*k3U, R0 + B41*k1R + B42*k2R + B43*k3R, A0 + B41*k1A + B42*k2A + B43*k3A, P0 + B41*k1P + B42*k2P + B43*k3P)
    
    # solve for F5 step
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray5 = first_r_derivative(A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A)
        Fint5 = solve_for_F_fast(G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P, dAdrarray5)
    else:
        Fint5 = F0
        
        
    k5G = tstep*kiG_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    #k5G[0] = 0.0
    k5U = tstep*kiU_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    #k5U[0] = 0.0
    k5R = tstep*kiR_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    k5A = tstep*kiA_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)
    k5P = tstep*kiP_function(r, Fint5, G0 + B51*k1G + B52*k2G + B53*k3G + B54*k4G, U0 + B51*k1U + B52*k2U + B53*k3U + B54*k4U, R0 + B51*k1R + B52*k2R + B53*k3R + B54*k4R, A0 + B51*k1A + B52*k2A + B53*k3A + B54*k4A, P0 + B21*k1P)

    # solve for F6
    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dAdrarray6 = first_r_derivative(A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A)
        Fint6 = solve_for_F_fast(G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P, dAdrarray6)
    else:
        Fint6 = F0
        
        
    k6G = tstep*kiG_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    #k6G[0] = 0.0
    k6U = tstep*kiU_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    #k6U[0] = 0.0
    k6R = tstep*kiR_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    k6A = tstep*kiA_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)
    k6P = tstep*kiP_function(r, Fint2, G0 + B61*k1G + B62*k2G + B63*k3G + B64*k4G + B65*k5G, U0 + B61*k1U + B62*k2U + B63*k3U + B64*k4U + B65*k5U, R0 + B61*k1R + B62*k2R + B63*k3R + B64*k4R + B65*k5R, A0 + B61*k1A + B62*k2A + B63*k3A + B64*k4A + B65*k5A, P0 + B61*k1P + B62*k2P + B63*k3P + B64*k4P + B65*k5P)

    
    # update with Runge-Kutta
    G1 = G0 + CH1*k1G + CH2*k2G + CH3*k3G + CH4*k4G + CH5*k5G + CH6*k6G
    U1 = U0 + CH1*k1U + CH2*k2U + CH3*k3U + CH4*k4U + CH5*k5U + CH6*k6U
    R1 = R0 + CH1*k1R + CH2*k2R + CH3*k3R + CH4*k4R + CH5*k5R + CH6*k6R
    A1 = A0 + CH1*k1A + CH2*k2A + CH3*k3A + CH4*k4A + CH5*k5A + CH6*k6A
    P1 = P0 + CH1*k1P + CH2*k2P + CH3*k3P + CH4*k4P + CH5*k5P + CH6*k6P
    
    #U1[0] = 0.0
    #G1[0] = 1.0
    
    truncation_error_G = max(abs(CT1*k1G + CT2*k2G + CT3*k3G + CT4*k4G + CT5*k5G + CT6*k6G))
    truncation_error_U = max(abs(CT1*k1U + CT2*k2U + CT3*k3U + CT4*k4U + CT5*k5U + CT6*k6U))
    truncation_error_R = max(abs(CT1*k1R + CT2*k2R + CT3*k3R + CT4*k4R + CT5*k5R + CT6*k6R))
    truncation_error_A = max(abs(CT1*k1A + CT2*k2A + CT3*k3A + CT4*k4A + CT5*k5A + CT6*k6A))
    truncation_error_P = max(abs(CT1*k1P + CT2*k2P + CT3*k3P + CT4*k4P + CT5*k5P + CT6*k6P))
    
    truncation_errors = np.array([truncation_error_G, truncation_error_U, truncation_error_R, truncation_error_A, truncation_error_P])
    #print(truncation_errors)
    
    hnewG = 0.9 * tstep * (acceptable_errors[0]/max(truncation_error_G, acceptable_errors[0]/1000))**(1/5)
    hnewU = 0.9 * tstep * (acceptable_errors[1]/max(truncation_error_U, acceptable_errors[1]/1000))**(1/5)
    hnewR = 0.9 * tstep * (acceptable_errors[2]/max(truncation_error_R, acceptable_errors[2]/1000))**(1/5)
    hnewA = 0.9 * tstep * (acceptable_errors[3]/max(truncation_error_A, acceptable_errors[3]/1000))**(1/5)
    hnewP = 0.9 * tstep * (acceptable_errors[4]/max(truncation_error_P, acceptable_errors[4]/1000))**(1/5)
    
    hnews = [hnewG, hnewU, hnewR, hnewA, hnewP]
    #print(hnews)
    hnew = min(hnews)
    hnewindex = hnews.index(hnew)
    
    if verbose == 1:
        #print("\n")
        print("TE problem variable: " + str(truncation_error_strings[hnewindex]))
        #print("\n")


    if truncation_errors[hnewindex] > acceptable_errors[hnewindex]:
        return matter_and_G_integrator_RKF45(r, F0, G0, U0, R0, A0, P0, indexflag, hnew, acceptable_errors_fraction, verbose);
    else:
        return [np.concatenate((U1, R1, A1, P1, G1)).astype(np.float64), hnew];







# MAKE ESTIMATES FOR INTEGRATION TIME
initialRvals = density_profile(rvals)
initialAvals = rvals*0.0
initialPvals = np.zeros(np.shape(rvals))
initialUvals = rvals*0.0

Gtimestart = time.time()
initialdAdrvals = first_r_derivative(initialAvals)
initialdRdrvals = first_r_derivative(initialRvals)
initialGvals = solve_for_G_fast(1, initialUvals, initialRvals, initialAvals, initialPvals, initialdAdrvals, initialdRdrvals)
Gtimeend = time.time()
Gtimetotal = Gtimeend - Gtimestart
initialdGdrvals = first_r_derivative(initialGvals)

Ftimestart = time.time()
initialFvals = solve_for_F_fast(initialGvals, initialUvals, initialRvals, initialAvals, initialPvals, initialdAdrvals)
Ftimeend = time.time()
Ftimetotal = Ftimeend - Ftimestart

MatterandGtimestart = time.time()
matterandGtest1 = matter_and_G_integrator(rvals, initialFvals, initialGvals, initialUvals, initialRvals, initialAvals, initialPvals, 1)
MatterandGtimeend = time.time()
MatterandGtimetotal = MatterandGtimeend - MatterandGtimestart

RKF45timestart = time.time()
matterandGtest2 = matter_and_G_integrator_RKF45(rvals, initialFvals, initialGvals, initialUvals, initialRvals, initialAvals, initialPvals, 1, dt, 0.001)
RKF45timeend = time.time()
RKF45timetotal = RKF45timeend - RKF45timestart

print("\n")

print("Total step time for RK4: " + str(Ftimetotal + MatterandGtimetotal))
print("Estimated total time: " + str((Ftimetotal + MatterandGtimetotal)*Nt/60) + " minutes")
print("For " + str(Nt) + " steps")
print("with time step dt = " + str(dt))
print("and r step dr = " + str(dr))

print("\n")

print("Total step time for RKF45: " + str(Ftimetotal + RKF45timetotal))
print("Estimated total time: " + str((Ftimetotal + RKF45timetotal)*Nt/60) + " minutes")
print("For " + str(Nt) + " steps")
print("with time step dt = " + str(dt))
print("and r step dr = " + str(dr))

print("\n")





# DO THE ACTUAL INTEGRATION
###################### Full Integrator RKF45 ########################
U0vals = 0.0*rvals
R0vals = density_profile(rvals) #* (1 + 1000 * np.exp(-(rvals - 10000)**2 / (2*1000**2)) / (np.sqrt(2*np.pi) * 1000))
A0vals = np.pi * (np.exp(-(rvals)**4 / (2*3600**4)) / (np.sqrt(2*np.pi) * 3600)) / (np.exp(-(rvals[0])**4 / (2*3600**4)) / (np.sqrt(2*np.pi) * 3600))
P0vals = np.zeros(np.shape(rvals))

metric0vals = metric_integrator(U0vals, R0vals, A0vals, P0vals)
F0vals = metric0vals[0:Nr]
G0vals = metric0vals[Nr:2*Nr]

# set up total arrays
FTotalvals, GTotalvals, UTotalvals, RTotalvals, ATotalvals, PTotalvals = np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr)), np.zeros((int(Nt/RESOLUTION), Nr))
FTotalvals[0, :] = F0vals
GTotalvals[0, :] = G0vals
UTotalvals[0, :] = U0vals
RTotalvals[0, :] = R0vals
ATotalvals[0, :] = A0vals
PTotalvals[0, :] = P0vals

F1vals = np.zeros(np.shape(F0vals))
G1vals = np.zeros(np.shape(G0vals))
U1vals = np.zeros(np.shape(U0vals))
R1vals = np.zeros(np.shape(R0vals))
A1vals = np.zeros(np.shape(A0vals))
P1vals = np.zeros(np.shape(P0vals))

print("\n")
print("STARTING ACTUAL INTEGRATOR")


if INTEGRATOR == "RK4":
    for i in range(1, Nt):
        matterandG1vals = matter_and_G_integrator(rvals, F0vals, G0vals, U0vals, R0vals, A0vals, P0vals, i)
        U1vals = matterandG1vals[0:Nr]
        R1vals = matterandG1vals[Nr:2*Nr]
        A1vals = matterandG1vals[2*Nr:3*Nr]
        P1vals = matterandG1vals[3*Nr:4*Nr]
        G1vals = matterandG1vals[4*Nr:5*Nr]
        F1vals = solve_for_F_fast(G1vals, U1vals, R1vals, A1vals, P1vals, first_r_derivative(A1vals))
	    
        F0vals = F1vals
        G0vals = G1vals
        U0vals = U1vals
        R0vals = R1vals
        A0vals = A1vals
        P0vals = P1vals
	    
        if int(i%RESOLUTION) == 0:
            FTotalvals[int(i/RESOLUTION),:] = F1vals
            GTotalvals[int(i/RESOLUTION),:] = G1vals
            UTotalvals[int(i/RESOLUTION),:] = U1vals
            RTotalvals[int(i/RESOLUTION),:] = R1vals
            ATotalvals[int(i/RESOLUTION),:] = A1vals
            PTotalvals[int(i/RESOLUTION),:] = P1vals
            print("\n")
            print("Plots for step ", str(i) + " of " + str(Nt) + " steps")
            print("Current time step:" + str(dt))
            print("Current physical time: " + str(i*dt))
            print("Fraction of evolution complete: " + str(i*dt/tfin))
            print("\n")



if INTEGRATOR == "RKF45":
    time_step = dt
    times = [0.0]
    for i in range(1, Nt):
        #print("step " + str(i) + " integration")
        matterandG1vals_and_timestep = matter_and_G_integrator_RKF45(rvals, F0vals, G0vals, U0vals, R0vals, A0vals, P0vals, i, time_step, 0.00001, i%RESOLUTION)
        time_step = matterandG1vals_and_timestep[1]
        times.append(times[-1] + time_step)
        matterandG1vals = matterandG1vals_and_timestep[0]
        U1vals = matterandG1vals[0:Nr]
        R1vals = matterandG1vals[Nr:2*Nr]
        A1vals = matterandG1vals[2*Nr:3*Nr]
        P1vals = matterandG1vals[3*Nr:4*Nr]
        G1vals = matterandG1vals[4*Nr:5*Nr]

        # integrate the metric sector
        F1vals = solve_for_F_fast(G1vals, U1vals, R1vals, A1vals, P1vals, first_r_derivative(A1vals))
        F0vals = F1vals
        G0vals = G1vals
        U0vals = U1vals
        R0vals = R1vals
        A0vals = A1vals
        P0vals = P1vals
	    
        if int(i%RESOLUTION) == 0:
            FTotalvals[int(i/RESOLUTION),:] = F1vals
            GTotalvals[int(i/RESOLUTION),:] = G1vals
            UTotalvals[int(i/RESOLUTION),:] = U1vals
            RTotalvals[int(i/RESOLUTION),:] = R1vals
            ATotalvals[int(i/RESOLUTION),:] = A1vals
            PTotalvals[int(i/RESOLUTION),:] = P1vals
            print("\n")
            print("Plots for step ", str(i) + " of " + str(Nt) + " steps")
            print("Current time step:" + str(time_step))
            print("Current physical time: " + str(times[-1]))
            print("Fraction of evolution complete: " + str(times[-1]/tfin))
            print("\n")


print("\n")
print("Integration Complete!")
print("\n")


# EXPORT DATA
np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-F" + ".txt", FTotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-G" + ".txt", GTotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-R" + ".txt", RTotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-U" + ".txt", UTotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-A" + ".txt", ATotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-P" + ".txt", PTotalvals, delimiter=",", comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa)))

if INTEGRATOR == "RKF45":
	np.savetxt(directory + "run-" + str(INTEGRATOR) + "-" + str(myindex) + "-time" + ".txt", times, comments="#", header=("epsilon: " + str(epsilon) + "  fa: " + str(fa) + "    resolution: " + str(RESOLUTION)))

print("Job Complete!")
