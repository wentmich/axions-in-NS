import numpy as np
from numpy import cos, sin, tan
from auxilliary_functions import *
from constants_GeV import *
from EOS import *
from TOV_initial_state import *
from metric_solver_radial_RK4 import *
from matter_solver_RK4 import *


def total_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    pressure = pow(G0,-2)*((pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*(-0.5 + 0.5*pow(U0,2)) + pow(G0,2)*(-1.*EOS(NRHO(nN0)) + epsilon*(0.00030789755994010005 - 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(U0,2)*(-0.00030789755994010005*epsilon - 1.*NRHO(nN0) + nN0*(0.05899999999999999 - 0.0417193000900063*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.00021771645254443624*epsilon*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))))*pow(-1. + pow(U0,2),-1)
    return pressure;

def total_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    density = pow(G0,-2)*((pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*(-0.5 + 0.5*pow(U0,2)) + pow(G0,2)*(-0.00030789755994010005*epsilon - 1.*NRHO(nN0) + nN0*(0.05899999999999999 - 0.0417193000900063*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.00021771645254443624*epsilon*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + pow(U0,2)*(0.00030789755994010005*epsilon - 1.*EOS(NRHO(nN0)) - 0.00021771645254443624*epsilon*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))))*pow(-1. + pow(U0,2),-1)
    return density;

def axion_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    pressure = pow(G0,-2)*((pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*(-0.5 + 0.5*pow(U0,2)) + pow(G0,2)*(pow(U0,2)*(nN0*(0.05899999999999999 - 0.0417193000900063*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + epsilon*(-0.00030789755994010005 + 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + epsilon*(0.00030789755994010005 - 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))))*pow(-1. + pow(U0,2),-1)
    return pressure;

def axion_angular_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    pressure = pow(G0,-2)*(-0.5*pow(dAdr,2)*pow(fa,2) + 0.5*pow(fa,2)*pow(P0,2) + epsilon*pow(G0,2)*(-0.00030789755994010005 + 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)))
    return pressure;
    
def baryon_pressure(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    pressure = (EOS(NRHO(nN0)) + NRHO(nN0)*pow(U0,2))*pow(1 - pow(U0,2),-1)
    return pressure;

def axion_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    density = pow(G0,-2)*((pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*(-0.5 + 0.5*pow(U0,2)) + pow(G0,2)*(nN0*(0.05899999999999999 - 0.0417193000900063*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + epsilon*(-0.00030789755994010005 + pow(U0,2)*(0.00030789755994010005 - 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.00021771645254443624*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))))*pow(-1. + pow(U0,2),-1)
    return density;

def baryon_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    density = (NRHO(nN0) + EOS(NRHO(nN0))*pow(U0,2))*pow(1 - pow(U0,2),-1)
    return density;
    

def kinetic_energy_density(rvals, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    density = (pow(fa,2)*pow(G0,-2)*pow(P0,2))/2. - (pow(U0,2)*(2*(EOS(NRHO(nN0)) + NRHO(nN0)) + 0.059*nN0*(-2 + pow(4 - 2*pow(beta,2) + 2*cos(A0)*pow(beta,2),0.5)))*pow(-1 + pow(U0,2),-1))/2.
    return density;


def axion_static_EOM_check(r, F0, G0, nN0, U0, A0, P0, epsilon, fa):
    dr = r[1] - r[0]
    dGdr = first_r_derivative(G0, dr)
    dFdr = first_r_derivative(F0, dr)
    dAdr = first_r_derivative(A0, dr)
    d2Adr2 = second_r_derivative(A0, dr)
    return -4*dAdr*dGdr*fa*pow(G0,-1) + 4*(d2Adr2*fa + dAdr*fa*(dFdr*pow(F0,-1) + 2*pow(r,-1))) - 0.00030789755994010005*(1.*epsilon - 191.62217463327138*nN0)*pow(beta,2)*pow(fa,-1)*pow(G0,2)*pow(1 - pow(beta,2)*pow(sin(A0/2.),2),-0.5)*sin(A0)

def div_baryon_current_check(r, F0, G0, nN0, U0, A0, P0, epsilon, fa, nNCUT):
    dr = r[1] - r[0]
    dGdr = first_r_derivative(G0, dr)
    dFdr = first_r_derivative(F0, dr)
    dAdr = first_r_derivative(A0, dr)
    d2Adr2 = second_r_derivative(A0, dr)
    dUdr = first_r_derivative(U0, dr)
    dnNdr = first_r_derivative(nN0, dr)
    num = -(F0*nN0*(4*fa*(EOS(NRHO(nN0)) + NRHO(nN0))*(1 + pow(U0,2))*(r*NRHO(nN0)*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) - 8.419468311620646e-38*dAdr*P0*NRHO(nN0)*pow(fa,2)*pow(r,2)*pow(U0,2) + EOS(NRHO(nN0))*(r*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) + U0*(2 - 1.6838936623241293e-37*NRHO(nN0)*pow(G0,2)*pow(r,2)) - 8.419468311620646e-38*dAdr*P0*pow(fa,2)*pow(r,2)*pow(U0,2)) + dnNdr*r*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(U0,3) - 8.419468311620646e-38*U0*pow(G0,2)*pow(r,2)*pow(EOS(NRHO(nN0)),2) - U0*(dnNdr*r*dEOSdrho(NRHO(nN0))*NMUN(nN0) - 2*NRHO(nN0) + 8.419468311620646e-38*pow(G0,2)*pow(r,2)*pow(NRHO(nN0),2)))*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) - 1.4654084596375735e-40*fa*U0*pow(G0,2)*pow(nN0,3)*pow(r,2)*(4*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(U0,2)*(4*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(beta,2)*(-2*pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + 0.059*(cos(2*A0)*pow(2,0.5)*pow(beta,4) - 4*cos(A0)*pow(beta,2)*(-5*pow(2,0.5) + pow(2,0.5)*pow(beta,2) + 3*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + (-8 + 3*pow(beta,2))*(-4*pow(2,0.5) + pow(2,0.5)*pow(beta,2) + 4*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)))) + 0.059*pow(nN0,2)*(-0.118*fa*r*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2))*(4*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(beta,2)*(-2*pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + fa*P0*r*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(U0,4)*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - pow(2,0.5)*pow(beta,2)*sin(A0)) + 0.059*U0*(4*fa*(-4*pow(2,0.5) + pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 4*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 5.051680986972388e-37*fa*(EOS(NRHO(nN0)) + NRHO(nN0))*pow(G0,2)*pow(r,2)*(4*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(beta,2)*(-2*pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) - dAdr*fa*r*pow(beta,2)*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))*sin(A0)) + pow(U0,3)*(1.6838936623241293e-37*fa*(EOS(NRHO(nN0)) + NRHO(nN0))*pow(G0,2)*pow(r,2)*(2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.059*(4*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(beta,2)*(-2*pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)))) - (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*(4*fa*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + dAdr*fa*r*pow(2,0.5)*pow(beta,2)*sin(A0))) + r*pow(U0,2)*(-2*dUdr*fa*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + fa*P0*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(-((1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + 0.059*(4*(pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + cos(A0)*pow(beta,2)*(2*pow(2,0.5) - pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + pow(beta,2)*(-2*pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)))) + 0.059*pow(beta,2)*(-pow(2,0.5) + pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))*sin(A0)))) + nN0*(0.236*fa*r*NRHO(nN0)*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2))*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 4*dnNdr*fa*r*dEOSdrho(NRHO(nN0))*(1 + dEOSdrho(NRHO(nN0)))*pow(U0,5)*pow(NMUN(nN0),2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) - 1.6838936623241293e-37*fa*U0*pow(G0,2)*pow(r,2)*pow(EOS(NRHO(nN0)),2)*(0.177*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 2*pow(U0,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) - fa*P0*r*NRHO(nN0)*pow(U0,4)*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 0.0834386001800126*pow(beta,2)*sin(A0)) - r*NRHO(nN0)*pow(U0,2)*(2*dUdr*fa*(2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + 0.059*(-2*pow(2,0.5) + pow(2,0.5)*pow(beta,2) - cos(A0)*pow(2,0.5)*pow(beta,2) + 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + fa*P0*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 0.0834386001800126*pow(beta,2)*sin(A0))) + pow(U0,3)*(-3.3677873246482585e-37*fa*pow(G0,2)*pow(r,2)*pow(NRHO(nN0),2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 2*dnNdr*fa*r*dEOSdrho(NRHO(nN0))*NMUN(nN0)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + NRHO(nN0)*(0.236*fa*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 8*fa*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + 0.0834386001800126*dAdr*fa*r*pow(beta,2)*sin(A0))) - 0.059*U0*(2*dnNdr*fa*r*dEOSdrho(NRHO(nN0))*NMUN(nN0)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 5.051680986972388e-37*fa*pow(G0,2)*pow(r,2)*pow(NRHO(nN0),2)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - NRHO(nN0)*(8*fa*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + dAdr*fa*r*pow(2,0.5)*pow(beta,2)*sin(A0))) + EOS(NRHO(nN0))*(0.236*fa*r*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2))*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - fa*P0*r*pow(U0,4)*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 0.0834386001800126*pow(beta,2)*sin(A0)) + pow(U0,3)*(0.236*fa*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 8*fa*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) - 6.735574649296517e-37*fa*NRHO(nN0)*pow(G0,2)*pow(r,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.0834386001800126*dAdr*fa*r*pow(beta,2)*sin(A0)) - 0.059*U0*(1.0103361973944776e-36*fa*NRHO(nN0)*pow(G0,2)*pow(r,2)*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 8*fa*(-2*pow(2,0.5) + pow(2,0.5)*pow(beta,2) - cos(A0)*pow(2,0.5)*pow(beta,2) + 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - dAdr*fa*r*pow(2,0.5)*pow(beta,2)*sin(A0)) - r*pow(U0,2)*(2*dUdr*fa*(2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + 0.059*(-2*pow(2,0.5) + pow(2,0.5)*pow(beta,2) - cos(A0)*pow(2,0.5)*pow(beta,2) + 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))) + fa*P0*(1.6838936623241293e-37*dAdr*r*pow(fa,2)*(0.059*(2*pow(2,0.5) - pow(2,0.5)*pow(beta,2) + cos(A0)*pow(2,0.5)*pow(beta,2) - 2*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 2*(1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) - 0.0834386001800126*pow(beta,2)*sin(A0)))))))
    den = 2*fa*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)*(-0.0834386001800126*nN0 + EOS(NRHO(nN0))*pow(2,0.5) + NRHO(nN0)*pow(2,0.5) + 0.059*nN0*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))*(-0.0834386001800126*nN0 + EOS(NRHO(nN0))*pow(2,0.5) + NRHO(nN0)*pow(2,0.5) + EOS(NRHO(nN0))*pow(2,0.5)*pow(U0,2) - nN0*NMUN(nN0)*pow(2,0.5)*pow(U0,2) - nN0*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(2,0.5)*pow(U0,2) + NRHO(nN0)*pow(2,0.5)*pow(U0,2) + 0.059*nN0*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5))
    ans = np.zeros(np.shape(r))
    np.divide(num.astype(np.float64), den.astype(np.float64), out=ans, where=(den > EOS(NRHO(nNCUT))))
    return ans;



def total_radial_energy_flux(rvals, F0, G0, nN0, U0, A0, P0, fa):
    dr = rvals[1] - rvals[0]
    dAdr = first_r_derivative(A0, dr)
    #ans = -(dAdr*F0*P0*pow(fa,2)*pow(G0,-3)) - (F0*U0*pow(G0,-1)*(2*(EOS(NRHO(nN0)) + NRHO(nN0)) + 0.059*nN0*(-2 + pow(4 - 2*pow(beta,2) + 2*cos(A0)*pow(beta,2),0.5)))*pow(-1 + pow(U0,2),-1))/2.
    ans = -(dAdr*P0*pow(F0,-1)*pow(fa,2)*pow(G0,-3)) - (U0*pow(F0,-1)*pow(G0,-1)*(2*(EOS(NRHO(nN0)) + NRHO(nN0)) + 0.059*nN0*(-2 + pow(4 - 2*pow(beta,2) + 2*cos(A0)*pow(beta,2),0.5)))*pow(-1 + pow(U0,2),-1))/2.
    return ans


def baryon_kinetic_energy_density(r, G0, nN0, U0):
    density = -((EOS(NRHO(nN0*pow(r,-2))) + NRHO(nN0*pow(r,-2)))*pow(U0,2)*pow(-1 + pow(U0,2),-1))
    return density;
