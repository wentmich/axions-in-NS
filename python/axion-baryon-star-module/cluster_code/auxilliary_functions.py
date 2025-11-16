import numpy as np
from numpy import cos, sin, tan

def sec(var):
    # secant function
    return 1.0 / cos(var);

def cot(var):
    # cotangent function
    return 1.0 / tan(var);

def smooth(arr, window_size):
    # smooths arr using a moving average with window_size points
    # returns array that's the same size as arr
    smoothed_arr = np.zeros_like(arr, dtype=float)
    boundary_offset = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - boundary_offset)
        end = min(len(arr), i + boundary_offset + 1)
        smoothed_arr[i] = np.mean(arr[start:end])
    
    return smoothed_arr;



#spatial derivative function with boundary conditions
def first_r_derivative(farray, dr):
    first_deriv = (-np.roll(farray, -2, 0) + 8.0 * np.roll(farray, -1, 0) - 8.0 * np.roll(farray, 1, 0) + np.roll(farray, 2, 0)) / (12.0 * dr)
    first_deriv[0] = (-3*farray[0] + 4*farray[1] - farray[2]) / (2*dr)#0.0
    first_deriv[1] = (-3*farray[1] + 4*farray[2] - farray[3]) / (2*dr)#(-farray[3] + (71.0/8.0)*farray[2] - (63.0/8.0)*farray[1])/(12.0*dr)
    first_deriv[-1] = (3*farray[-1] - 4*farray[-2] + farray[-3]) / (2*dr)
    first_deriv[-2] = (3*farray[-2] - 4*farray[-3] + farray[-4]) / (2*dr)
    return first_deriv;

def second_r_derivative(farray, dr):
    second_deriv = (-np.roll(farray, -2, 0) + 16.0 * np.roll(farray, -1, 0) - 30.0 * farray + 16.0 * np.roll(farray, 1, 0) - np.roll(farray, 2, 0)) / (12.0 * dr**2)
    second_deriv[0] = (2*farray[0] - 5*farray[1] + 4*farray[2] - farray[3]) / (dr**2)#(-farray[1] + farray[2]) / (16.0*dr**2)
    second_deriv[1] = (2*farray[1] - 5*farray[2] + 4*farray[3] - farray[4]) / (dr**2)#(-(105.0/8.0)*farray[1] + (107.0/8.0)*farray[2] - farray[3]) / (12.0*dr**2)
    second_deriv[-1] = (2*farray[-1] - 5*farray[-2] + 4*farray[-3] - farray[-4]) / (dr**2)
    second_deriv[-2] = (2*farray[-2] - 5*farray[-3] + 4*farray[-4] - farray[-5]) / (dr**2)
    return second_deriv;


def enforce_rc_BC(farray, ZEROATRC=0):
    if ZEROATRC == 1:
        ans = farray
        ans[0] = 0.0
        ans[1] = ans[2] / 9.0
    else: 
        ans = farray
        ans[0] = (9.0 * ans[1] - ans[2]) / 8.0
    return ans;



def L2_norm(rvals, radial_data, end_index = -1):
    dr = rvals[1] - rvals[0]
    ans = np.sum(4*np.pi*rvals[:end_index]**2 * radial_data[:end_index]**2 * dr)
    return ans


def zero_crossings_indices(arr, min_index=0):
    arr = np.asarray(arr)
    signs = np.sign(arr)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    return sign_changes[sign_changes > min_index]
