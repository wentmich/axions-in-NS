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
import scipy.signal as sig
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

RESOLUTION = 1

def get_frequencies_1D_array(data, times, tstep_start, tstep_end):
    # returns the FFT amplitudes for data with frequencies obtained from time values in times
    dataFFT = np.fft.rfft(data[tstep_start:tstep_end])
    Nt = int(len(times[tstep_start*RESOLUTION:tstep_end*RESOLUTION]) / RESOLUTION)
    dt = times[RESOLUTION] - times[0]

    angular_frequencies = np.fft.rfftfreq(Nt, dt) # gives frequencies in GeV
  
    print(np.max(np.abs(dataFFT)))
    print(angular_frequencies[1] - angular_frequencies[0]) 
    return [angular_frequencies, np.abs(dataFFT)];

def get_frequencies_1D(data, times, index, tstep_start, tstep_end):
    # returns the FFT amplitudes for data with frequencies obtained from time values in times
    dataFFT = np.fft.rfft(data[tstep_start:tstep_end, index])
    Nt = int(len(times[tstep_start*RESOLUTION:tstep_end*RESOLUTION]) / RESOLUTION)
    dt = times[RESOLUTION] - times[0]

    angular_frequencies = np.fft.rfftfreq(Nt, dt) # gives frequencies in GeV
  
    print(np.max(np.abs(dataFFT)))
    print(angular_frequencies[1] - angular_frequencies[0]) 
    return [angular_frequencies, np.abs(dataFFT)];


def get_fft_peaks_and_widths(fs, As):
    # finds the peaks of the amplutudes in As at frequencies in fs
    # returns the peak frequencies, the peak amplitudes, and the estimated full-width, half-max values of the peaks
    peak_indices = sig.find_peaks(As, distance=20, width=0.1)#, prominence=0.1)
    df = fs[1] - fs[0]
    peak_frequencies = []
    peak_amplitudes  = []
    peak_widths = []
    peak_width_heights = []
    peak_width_fractional_heights = []
    fwhms = []
    for i in range(len(peak_indices[0])):
        peak_frequencies.append(fs[peak_indices[0][i]])
        peak_amplitudes.append(As[peak_indices[0][i]])
        peak_widths.append(peak_indices[1]["widths"][i]*df)
        peak_width_heights.append(peak_indices[1]["width_heights"][i])
        peak_width_fractional_heights.append(peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]])
        fwhms.append((peak_indices[1]["widths"][i]*df)*(peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]]) / (1 - (peak_indices[1]["width_heights"][i] / As[peak_indices[0][i]])))
    
    return [peak_frequencies, peak_amplitudes, fwhms];


def total_frequency_analysis_function(directory):
    # returns the axion peak locations and amplitudes and the jr peaks and amplitudes
    NSmasses = np.load(directory + "/NSmasses.npy")
    Ftotal = np.load(directory + "/F-solution.npy", allow_pickle=False)
    Gtotal = np.load(directory + "/G-solution.npy", allow_pickle=False)
    Atotal = np.load(directory + "/A-solution.npy", allow_pickle=False)
    nNtotal = np.load(directory + "/nN-solution.npy", allow_pickle=False)
    Utotal = np.load(directory + "/U-solution.npy", allow_pickle=False)
    #Ptotal = np.load(directory + "/P-solution.npy", allow_pickle=False)
    times  = np.load(directory + "/times.npy", allow_pickle=False)
    radii  = np.load(directory + "/radii.npy", allow_pickle=False)
    rvals = np.copy(radii)

    dr = radii[1] - radii[0]

    kinetic_energy_densities = np.zeros(np.shape(Gtotal))
    total_kinetic_energies = np.zeros(np.shape(Gtotal)[0])
    for i in range(np.shape(Gtotal)[0]):
        kinetic_energy_densities[i, :] = baryon_kinetic_energy_density(rvals, Gtotal[i, :], nNtotal[i, :], Utotal[i, :])
        total_kinetic_energies[i] = np.sum(kinetic_energy_densities[i,:]*dr*rvals**2*4*np.pi*Gtotal[i,:]*Ftotal[i,:])

    Ntsuccessful = int(len(NSmasses) / RESOLUTION - 10)

    Nr = len(radii)
    dr = radii[1] - radii[0]

    # get frequency data from Fourier Transform
    axion_frequency_20km = get_frequencies_1D(Atotal, times, 1500, 1000, Ntsuccessful)
    KE_frequency = get_frequencies_1D_array(total_kinetic_energies, times, 0, Ntsuccessful)
#    jr_frequency_100m = get_frequencies_1D(Utotal*nNtotal / (np.sqrt(1 - Utotal**2)*Gtotal), times, 10, 0, Ntsuccessful)
#    jr_frequency_100m = get_frequencies_1D(Utotal*baryon_density(radii, Gtotal, nNtotal, Utotal, Atotal, Atotal, 1e-1, 1e16) / (np.sqrt(1 - Utotal**2)*Gtotal), times, 10, 0, Ntsuccessful)
    jr_frequency_100m = get_frequencies_1D(baryon_density(radii, Gtotal, nNtotal, Utotal, Atotal, Atotal, 1e-1, 1e16), times, 10, 0    , Ntsuccessful)

    # get frequency peaks
    axion_peak_data = get_fft_peaks_and_widths(axion_frequency_20km[0], axion_frequency_20km[1])
    jr_peak_data = get_fft_peaks_and_widths(jr_frequency_100m[0], jr_frequency_100m[1])

    return [axion_peak_data[0], axion_peak_data[1], jr_peak_data[0], jr_peak_data[1], axion_frequency_20km[0], axion_frequency_20km[1], jr_frequency_100m[0], jr_frequency_100m[1], KE_frequency[0], KE_frequency[1]];
