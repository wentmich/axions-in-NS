import numpy as np
from numpy import cos, sin, tan, roll
from auxilliary_functions import *
from constants_GeV import *
from EOS import *
from TOV_initial_state import *
from metric_solver_radial_RK4 import *
from matter_solver_RK4_frictest import *

def matter_and_metric_solver_frictest(rvals, Fv, Gv, nNv, Uv, Av, Pv, epsilon, fa, dt, nNCUT, linfric, densfric):
    Nr = len(rvals)
    newvals = matter_integrator(rvals, Fv, Gv, nNv, Uv, Av, Pv, epsilon, fa, dt, nNCUT, linfric, densfric)
    nNv, Uv, Av, Pv = newvals[:Nr], newvals[Nr:2*Nr], newvals[2*Nr:3*Nr], newvals[3*Nr:4*Nr]
#    nNvzeros = np.zeros(np.shape(nNv))
#    np.multiply(nNv.astype(np.float64), nNvzeros.astype(np.float64), out=nNv, where=(nNv < nNCUT))
#    np.multiply(Uv.astype(np.float64), nNvzeros.astype(np.float64), out=Uv, where=(nNv < nNCUT))
    Gv = solve_G_constraint_fast(rvals, nNv, Uv, Av, Pv, epsilon, fa)
    Fv = solve_F_constraint_fast(rvals, Gv, nNv, Uv, Av, Pv, epsilon, fa)

    return np.concatenate((Fv, Gv, nNv, Uv, Av, Pv));
