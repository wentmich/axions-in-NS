import numpy as np
from numpy import cos, tan, sin
from auxilliary_functions import *
from constants_GeV import *
from EOS import *


# step forward the time dependent parameters
def sponge_function(r):
    dr = r[1] - r[0]
    Nr = len(r)
    return np.heaviside(r - r[Nr - NFRICTIONSITES], 1) * ((r - r[Nr - NFRICTIONSITES]) / (dr*FRICTIONSITESCALE))**3 / np.sqrt(1 + ((r - r[Nr - NFRICTIONSITES])/ (dr*FRICTIONSITESCALE))**6);


def Q_fnc_dUdt(U0, cut_off_curvature):
    if VISCOSITYORDER == 2:
        ans = np.roll(U0, -1, 0) - 2.0*U0 + np.roll(U0, 1, 0)
        ans[0]  = U0[0] - 2.0*U0[0] + U0[1]
        ans[-1] = U0[-3] - 2.0*U0[-2] + U0[-1]

        Ccut = cut_off_curvature[0]
        Cexp = cut_off_curvature[1]
        Scut = (np.minimum(np.abs(ans), np.ones(np.shape(ans))*Ccut) / Ccut)**Cexp

    if VISCOSITYORDER == 4:
        ans = np.roll(U0, -2, 0) - 4.0*np.roll(U0, -1, 0) + 6.0*U0 - 4.0*np.roll(U0, 1, 0) + np.roll(U0, 2, 0)
        ans[0]  = U0[0] - 4.0*U0[1] + 6.0*U0[2] - 4.0*U0[3] + U0[4]
        ans[1]  = U0[0] - 4.0*U0[1] + 6.0*U0[2] - 4.0*U0[3] + U0[4]
        ans[-1]  = U0[-5] - 4.0*U0[-4] + 6.0*U0[-3] - 4.0*U0[-2] + U0[-1]
        ans[-2]  = U0[-5] - 4.0*U0[-4] + 6.0*U0[-3] - 4.0*U0[-2] + U0[-1]
    
        Ccut = cut_off_curvature[0]
        Cexp = cut_off_curvature[1]
        Scut = (-1.0)*(np.minimum(np.abs(ans), np.ones(np.shape(ans))*Ccut) / Ccut)**Cexp

    return 0.5*ans*Scut 


#def Q_fnc_dUdt(U0, cut_off_curvature):
#    ans = np.roll(U0, -1, 0) - 2.0*U0 + np.roll(U0, 1, 0)
#    ans[0]  = U0[0] - 2.0*U0[0] + U0[1]
#    ans[-1] = U0[-3] - 2.0*U0[-2] + U0[-1]

#    Ccut = cut_off_curvature[0]
#    Cexp = cut_off_curvature[1]
#    Scut = (np.minimum(np.abs(ans), np.ones(np.shape(ans))*Ccut) / Ccut)**Cexp

#    return 0.5*ans*Scut 

def dnNdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa):
    dr = r[1] - r[0]
    ans1 = np.zeros(np.shape(r))
    num1 = F0*(-2*fa*r*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2))*(EOS(NRHO(nN0)) + NRHO(nN0) + 0.059*nN0*(-1 + pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)))*pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5) + 1.6838936623241293e-37*dAdr*P0*pow(fa,3)*pow(r,2)*pow(U0,2)*(EOS(NRHO(nN0)) + NRHO(nN0) + 0.059*nN0*(-1 + pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)))*pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5) + 2*fa*U0*pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)*(2*(EOS(NRHO(nN0)) + NRHO(nN0))*(-1 + 4.209734155810323e-38*(EOS(NRHO(nN0)) + NRHO(nN0))*pow(G0,2)*pow(r,2)) + dnNdr*r*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0) - 0.059*pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)) + 0.118*nN0*(-1 + 8.419468311620646e-38*(EOS(NRHO(nN0)) + NRHO(nN0))*pow(G0,2)*pow(r,2))*(-1 + pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)) + 2.930816919275147e-40*pow(G0,2)*pow(nN0,2)*pow(r,2)*pow(-1 + pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5),2)) - 0.0295*dAdr*fa*nN0*r*U0*pow(beta,2)*pow(pow(sin(A0),2),0.5) - 0.0295*fa*nN0*P0*r*pow(beta,2)*pow(U0,2)*pow(pow(sin(A0),2),0.5))
    den1 = 2*fa*G0*r*(NMUN(nN0)*(1 - dEOSdrho(NRHO(nN0))*pow(U0,2)) + 0.059*(-1 + pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)))*pow(1 + ((-1 + cos(A0))*pow(beta,2))/2.,0.5)
    np.divide(num1.astype(np.float64), den1.astype(np.float64), out=ans1, where=(abs(den1) > 1e-100))

    nN0LF = smooth(nN0, 2)
    nN0HF = nN0 - nN0LF
    ans1 = ans1 + (0.1/5.06773e16)*(dr/5.06773e16)**(0) * Q_fnc_dUdt(nN0HF, [1e-8, 1])/1.0

    return enforce_rc_BC(ans1).astype(np.float64)

def Faxion(A0):
    return np.sqrt(1 - beta**2 * np.sin(A0 / 2)**2);

def dUdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa, nNCUT):
    dr = r[1] - r[0]
    ans1 = np.zeros(np.shape(r))
    ans2 = np.zeros(np.shape(r))
    fofa0 = Faxion(A0)
    num1 = F0*(1.804782756151958e-43*epsilon*fa*nN0*pow(fofa0,4)*pow(G0,2)*pow(r,2)*(-1 + pow(U0,2)) + 0.059*fa*pow(fofa0,3)*(5.184667498281982e-41*epsilon*(EOS(NRHO(nN0)) + NRHO(nN0))*pow(G0,2)*pow(r,2)*(-1 + pow(U0,2)) + 9.934972607712362e-39*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(G0,2)*pow(nN0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(U0,2) + nN0*(-0.118*dUdr*r*U0 - (-1 + pow(U0,2))*(0.059 - 4.967486303856181e-39*(2*dAdr*P0*U0*pow(fa,2) + pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2) + pow(G0,2)*(0.059*(-1 + 1.5554002494845948e-40*epsilon*pow(r,2)) + 1.6838936623241293e-37*pow(r,2)*(-0.059*EOS(NRHO(nN0)) + 0.00030789755994010005*epsilon*NMUN(nN0)*(-1 + dEOSdrho(NRHO(nN0))*pow(U0,2))))))) - fa*pow(fofa0,2)*(0.118*dnNdr*r*dEOSdrho(NRHO(nN0))*NMUN(nN0) - 0.059*NRHO(nN0) + 0.118*dUdr*r*U0*NRHO(nN0) + 0.059*NRHO(nN0)*pow(G0,2) + 9.934972607712362e-39*dAdr*P0*U0*NRHO(nN0)*pow(fa,2)*pow(r,2) + 4.967486303856181e-39*NRHO(nN0)*pow(dAdr,2)*pow(fa,2)*pow(r,2) - 6.117907647972739e-42*epsilon*NRHO(nN0)*pow(G0,2)*pow(r,2) + 5.184667498281982e-41*epsilon*NMUN(nN0)*NRHO(nN0)*pow(G0,2)*pow(r,2) + 4.967486303856181e-39*NRHO(nN0)*pow(fa,2)*pow(P0,2)*pow(r,2) - 0.236*dnNdr*r*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(U0,2) + 0.059*NRHO(nN0)*pow(U0,2) - 0.059*NRHO(nN0)*pow(G0,2)*pow(U0,2) - 4.967486303856181e-39*NRHO(nN0)*pow(dAdr,2)*pow(fa,2)*pow(r,2)*pow(U0,2) + 6.117907647972739e-42*epsilon*NRHO(nN0)*pow(G0,2)*pow(r,2)*pow(U0,2) - 5.184667498281982e-41*epsilon*NMUN(nN0)*NRHO(nN0)*pow(G0,2)*pow(r,2)*pow(U0,2) - 5.184667498281982e-41*epsilon*dEOSdrho(NRHO(nN0))*NMUN(nN0)*NRHO(nN0)*pow(G0,2)*pow(r,2)*pow(U0,2) - 4.967486303856181e-39*NRHO(nN0)*pow(fa,2)*pow(P0,2)*pow(r,2)*pow(U0,2) + 1.1723267677100588e-39*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(G0,2)*pow(nN0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(U0,2) + EOS(NRHO(nN0))*(0.118*dUdr*r*U0 - 0.059*(-1 + 8.419468311620646e-38*(2*dAdr*P0*U0*pow(fa,2) + pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))*(-1 + pow(U0,2)) + pow(G0,2)*(-1 + pow(U0,2))*(0.059*(-1 + 1.0369334996563964e-40*epsilon*pow(r,2)) + 1.6838936623241293e-37*pow(r,2)*(-0.059*NRHO(nN0) + 0.00030789755994010005*epsilon*NMUN(nN0)*(-1 + dEOSdrho(NRHO(nN0))*pow(U0,2))))) - 9.934972607712362e-39*dAdr*P0*NRHO(nN0)*pow(fa,2)*pow(r,2)*pow(U0,3) + 0.118*dnNdr*r*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(U0,4) + 5.184667498281982e-41*epsilon*dEOSdrho(NRHO(nN0))*NMUN(nN0)*NRHO(nN0)*pow(G0,2)*pow(r,2)*pow(U0,4) - 0.059*nN0*(2*r*U0*(0.118 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) + (0.118 - NMUN(nN0))*(-1 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2)) + (0.118 - NMUN(nN0) + 3*dEOSdrho(NRHO(nN0))*NMUN(nN0) + 8.419468311620646e-38*(-0.118 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 8.419468311620646e-38*(-0.118 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,2) + pow(G0,2)*(-1 + pow(U0,2))*(0.118*(-1 + 7.777001247422974e-41*epsilon*pow(r,2) - 1.6838936623241293e-37*EOS(NRHO(nN0))*pow(r,2)) + NMUN(nN0)*(1 - 1.0369334996563964e-40*epsilon*pow(r,2) + 1.6838936623241293e-37*EOS(NRHO(nN0))*pow(r,2) + dEOSdrho(NRHO(nN0))*(-1 + 1.0369334996563964e-40*epsilon*pow(r,2) + 1.6838936623241293e-37*(EOS(NRHO(nN0)) + 2*NRHO(nN0))*pow(r,2))*pow(U0,2))) + 1.6838936623241293e-37*dAdr*P0*(-0.118 - (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(r,2)*pow(U0,3) - dEOSdrho(NRHO(nN0))*NMUN(nN0)*(3 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))*pow(U0,4)) - 9.934972607712362e-39*pow(G0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(EOS(NRHO(nN0)),2)) + 0.059*nN0*r*(dAdr*fa + fa*P0*U0)*(0.059 - NMUN(nN0))*pow(beta,2)*(-1 + pow(U0,2))*pow(-(pow(beta,-4)*(-1 + pow(fofa0,2))*(-1 + pow(beta,2) + pow(fofa0,2))),0.5) + fofa0*(5.861633838550294e-40*fa*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(G0,2)*pow(nN0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(U0,2) + 0.059*nN0*(-(fa*(0.059 - NMUN(nN0))*(-1 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))) + fa*pow(G0,2)*(-1 + pow(U0,2))*(0.059 - 3.0589538239863696e-42*epsilon*pow(r,2) + 9.934972607712362e-39*EOS(NRHO(nN0))*pow(r,2) - NMUN(nN0)*(1 - 5.184667498281982e-41*epsilon*pow(r,2) + 1.6838936623241293e-37*EOS(NRHO(nN0))*pow(r,2) + dEOSdrho(NRHO(nN0))*(-1 + 5.184667498281982e-41*epsilon*pow(r,2) + 1.6838936623241293e-37*(EOS(NRHO(nN0)) + 2*NRHO(nN0))*pow(r,2))*pow(U0,2))) + fa*dEOSdrho(NRHO(nN0))*NMUN(nN0)*(3 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))*pow(U0,4) + 0.059*dAdr*fa*r*pow(beta,2)*pow(-(pow(beta,-4)*(-1 + pow(fofa0,2))*(-1 + pow(beta,2) + pow(fofa0,2))),0.5) + fa*P0*r*pow(U0,3)*(1.6838936623241293e-37*dAdr*r*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2) - 0.059*pow(beta,2)*pow(-(pow(beta,-4)*(-1 + pow(fofa0,2))*(-1 + pow(beta,2) + pow(fofa0,2))),0.5)) + r*U0*(-2*fa*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) + 0.059*fa*P0*pow(beta,2)*pow(-(pow(beta,-4)*(-1 + pow(fofa0,2))*(-1 + pow(beta,2) + pow(fofa0,2))),0.5)) + pow(U0,2)*(0.059*fa*(-1 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2)) + fa*NMUN(nN0)*(1 - 3*dEOSdrho(NRHO(nN0)) - 8.419468311620646e-38*(1 + dEOSdrho(NRHO(nN0)))*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2)) - 0.059*dAdr*fa*r*pow(beta,2)*pow(-(pow(beta,-4)*(-1 + pow(fofa0,2))*(-1 + pow(beta,2) + pow(fofa0,2))),0.5))) + fa*(NRHO(nN0)*(2*r*U0*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) + (0.059 - NMUN(nN0))*(-1 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2)) + (0.059 - NMUN(nN0) + 3*dEOSdrho(NRHO(nN0))*NMUN(nN0) + 8.419468311620646e-38*(-0.059 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 8.419468311620646e-38*(-0.059 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,2) + pow(G0,2)*(-1 + 5.184667498281982e-41*epsilon*pow(r,2))*(-1 + pow(U0,2))*(0.059 + NMUN(nN0)*(-1 + dEOSdrho(NRHO(nN0))*pow(U0,2))) - 1.6838936623241293e-37*dAdr*P0*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(r,2)*pow(U0,3) - dEOSdrho(NRHO(nN0))*NMUN(nN0)*(3 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))*pow(U0,4)) + EOS(NRHO(nN0))*(2*r*U0*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*(dUdr + 8.419468311620646e-38*dAdr*P0*r*pow(fa,2)) + (0.059 - NMUN(nN0))*(-1 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2)) + (0.059 - NMUN(nN0) + 3*dEOSdrho(NRHO(nN0))*NMUN(nN0) + 8.419468311620646e-38*(-0.059 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(dAdr,2)*pow(fa,2)*pow(r,2) + 8.419468311620646e-38*(-0.059 + (1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(P0,2)*pow(r,2))*pow(U0,2) + pow(G0,2)*(-1 + pow(U0,2))*((-1 + 5.184667498281982e-41*epsilon*pow(r,2))*(0.059 + NMUN(nN0)*(-1 + dEOSdrho(NRHO(nN0))*pow(U0,2))) + 1.6838936623241293e-37*NRHO(nN0)*pow(r,2)*(-0.059 + NMUN(nN0)*(1 + dEOSdrho(NRHO(nN0))*pow(U0,2)))) - 1.6838936623241293e-37*dAdr*P0*(0.059 + (-1 + dEOSdrho(NRHO(nN0)))*NMUN(nN0))*pow(fa,2)*pow(r,2)*pow(U0,3) - dEOSdrho(NRHO(nN0))*NMUN(nN0)*(3 + 8.419468311620646e-38*(pow(dAdr,2)*pow(fa,2) + pow(fa,2)*pow(P0,2))*pow(r,2))*pow(U0,4)) - 1.6838936623241293e-37*(0.059 - NMUN(nN0))*pow(G0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(EOS(NRHO(nN0)),2) + 1.6838936623241293e-37*dEOSdrho(NRHO(nN0))*NMUN(nN0)*pow(G0,2)*pow(r,2)*(-1 + pow(U0,2))*pow(U0,2)*pow(NRHO(nN0),2) + 2*dnNdr*r*dEOSdrho(NRHO(nN0))*(0.059 - NMUN(nN0))*NMUN(nN0)*pow(-1 + pow(U0,2),2))))
    den1 = 2*fa*fofa0*G0*r*(0.059*(-1 + fofa0)*nN0 + EOS(NRHO(nN0)) + NRHO(nN0))*(0.059*(-1 + fofa0) + NMUN(nN0)*(1 - dEOSdrho(NRHO(nN0))*pow(U0,2)))
    #######
    num2 = nN0
    den2 = nN0
    np.divide(num1.astype(np.float64), den1.astype(np.float64), out=ans1, where=(abs(den1) > 1e-100))
    np.divide(num2.astype(np.float64), den2.astype(np.float64), out=ans2, where=(num2 > nsatinGeV3*1e-10))
    ZETAFRICTION = 0.5 * sponge_function(r) / dr
    ans1 = ans1 + (0.5/5.06773e16)*(dr/5.06773e16)**(0) *Q_fnc_dUdt(U0, [1e-15, 1])/1.0 - (1.0e-3/5.06773e16) * U0 #- ZETAFRICTION*U0 #- 5e-3*(nN0/nN0[0])**2 * U0 / dr 
    ans1 = ans1*ans2
    if np.isnan(ans1).any():
        print("got nan in U")
    ansnonan = np.nan_to_num(ans1)
    return enforce_rc_BC(ansnonan, ZEROATRC=1).astype(np.float64)

def dAdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa):
    dr = r[1] - r[0]
    #ans = F0*P0*pow(G0,-1)
    ans = F0*P0*pow(G0,-1)
    A0LF = smooth(A0, 2)
    A0HF = A0 - A0LF
    ans = ans + (0.1/5.06773e16)*(dr/5.06773e16)**(0) *Q_fnc_dUdt(A0HF, [1e-12, 1])/1.0
    return enforce_rc_BC(ans).astype(np.float64)

def dPdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa):
    dr = r[1] - r[0]
    #ZETAFRICTION = np.ones(np.shape(nN0)) * 1e-19 + sponge_function(r)*1e-16 + 1e-18 * nN0**2 / nsatinGeV3**2
    ZETAFRICTION = 0.5 * sponge_function(r) / (2*dr)
    ans = (F0*pow(G0,-1)*pow(r,-1)*(dAdr*fa + d2Adr2*fa*r + pow(fa,-1)*pow(G0,2)*pow(1 - pow(beta,2)*pow(sin(A0/2.),2),-0.5)*(dAdr*pow(fa,2)*(5.184667498281982e-41*epsilon*pow(r,2) - 2.592333749140991e-41*epsilon*pow(beta,2)*pow(r,2) + 2.592333749140991e-41*epsilon*cos(A0)*pow(beta,2)*pow(r,2) - 3.6661135462326824e-41*epsilon*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + 5.953463137132212e-38*EOS(NRHO(nN0))*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) - 5.953463137132212e-38*NRHO(nN0)*pow(r,2)*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5) + pow(1 - pow(beta,2)*pow(sin(A0/2.),2),0.5)) - 0.00007697438998502501*epsilon*r*pow(beta,2)*sin(A0) + nN0*r*(dAdr*r*pow(fa,2)*(-4.967486303856181e-39 + 2.4837431519280904e-39*pow(beta,2) - 2.4837431519280904e-39*cos(A0)*pow(beta,2) + 3.5125432509080046e-39*pow(2 - pow(beta,2) + cos(A0)*pow(beta,2),0.5)) + 0.01475*pow(beta,2)*sin(A0))))) / fa - ZETAFRICTION * P0
    ans = ans + (0.1/5.06773e16)*(dr/5.06773e16)**(0) *Q_fnc_dUdt(P0, [1e-23, 1])/1.0 - ZETAFRICTION*P0
    return enforce_rc_BC(ans).astype(np.float64)

def dTimeDependentFunctionsdt_for_RK4(r, F0, dynamicVec, epsilon, fa, dr, Nr, nNCUT):
    G0 = dynamicVec[:Nr]
    nN0 = dynamicVec[Nr:2*Nr]
    U0 = dynamicVec[2*Nr:3*Nr]
    A0 = dynamicVec[3*Nr:4*Nr]
    P0 = dynamicVec[4*Nr:5*Nr]
    
    dAdr = first_r_derivative(A0, dr)
    d2Adr2 = second_r_derivative(A0, dr)
    dnNdr = first_r_derivative(nN0, dr)
    dUdr = first_r_derivative(U0, dr)
    
    dG = dGdt_for_RK4(r, F0, G0, nN0, U0, dAdr, P0, epsilon, fa)
    dN = dnNdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    dU = dUdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa, nNCUT)
    dA = dAdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    dP = dPdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    
    return np.concatenate((dG, dN, dU, dA, dP)).astype(np.float64)

def dMatterdt_for_RK4(r, F0, G0, dynamicVec, epsilon, fa, dr, Nr, nNCUT):
    nN0 = dynamicVec[0:Nr]
    U0 = dynamicVec[Nr:2*Nr]
    A0 = dynamicVec[2*Nr:3*Nr]
    P0 = dynamicVec[3*Nr:4*Nr]
    
    dAdr = first_r_derivative(A0, dr)
    d2Adr2 = second_r_derivative(A0, dr)
    dnNdr = first_r_derivative(nN0, dr)
    dUdr = first_r_derivative(U0, dr)
    
    dN = dnNdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    dU = dUdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa, nNCUT)
    dA = dAdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    dP = dPdt_for_RK4(r, F0, G0, nN0, dnNdr, U0, dUdr, A0, dAdr, d2Adr2, P0, epsilon, fa)
    
    return np.concatenate((dN, dU, dA, dP)).astype(np.float64)


# integrator
def matter_and_G_integrator(rvals, F0, G0, nN0, U0, A0, P0, epsilon, fa, dt):
    dr = rvals[1] - rvals[0]
    Nr = len(rvals)
    
    dvec1 = np.concatenate((G0, nN0, U0, A0, P0))
    k1tot = dTimeDependentFunctionsdt_for_RK4(rvals, F0, dvec1, epsilon, fa, dr, Nr)
    k2tot = dTimeDependentFunctionsdt_for_RK4(rvals, F0, dvec1 + k1tot*dt/2, epsilon, fa, dr, Nr)
    k3tot = dTimeDependentFunctionsdt_for_RK4(rvals, F0, dvec1 + k2tot*dt/2, epsilon, fa, dr, Nr)
    k4tot = dTimeDependentFunctionsdt_for_RK4(rvals, F0, dvec1 + k3tot*dt, epsilon, fa, dr, Nr)
    
    dvec2 = dvec1 + (1/6)*k1tot*dt + (1/3)*k2tot*dt + (1/3)*k3tot*dt + (1/6)*k4tot*dt
    
    # make sure functions are all flat across the origin
    dvec2[0]   = (9.0*dvec2[1] - dvec2[2]) / 8.0
    #dvec2[Nr]  = 0.0
    dvec2[Nr]  = (9.0*dvec2[Nr+1] - dvec2[Nr+2]) / 8.0
    #dvec2[Nr + 1] = dvec2[Nr + 2] / 9.0
    dvec2[2*Nr] = (9.0*dvec2[2*Nr+1] - dvec2[2*Nr+2]) / 8.0
    dvec2[3*Nr] = (9.0*dvec2[3*Nr+1] - dvec2[3*Nr+2]) / 8.0
    dvec2[4*Nr] = (9.0*dvec2[4*Nr+1] - dvec2[4*Nr+2]) / 8.0
    
    return dvec2.astype(np.float64)

def matter_integrator(rvals, F0, G0, nN0, U0, A0, P0, epsilon, fa, dt, nNCUT):
    dr = rvals[1] - rvals[0]
    Nr = len(rvals)
    
    dvec1 = np.concatenate((nN0, U0, A0, P0))
    k1tot = dMatterdt_for_RK4(rvals, F0, G0, dvec1, epsilon, fa, dr, Nr, nNCUT)

    if INTERMEDIATEMETRICINTEGRATOR == 1:
        dvec1h = dvec1 + k1tot*dt/2.0
        nNvh, Uvh,  Avh, Pvh = dvec1h[:Nr], dvec1h[Nr:2*Nr], dvec1h[2*Nr:3*Nr], dvec1h[3*Nr:]
        Gh = solve_G_constraint_fast(rvals, nNvh, Uvh, Avh, Pvh, epsilon, fa) 
        Fh = solve_F_constraint_fast(rvals, Gvh, nNvh, Uvh, Avh, Pvh, epsilon, fa) 
        k2tot = dMatterdt_for_RK4(rvals, Fh, Gh, dvec1 + k1tot*dt/2, epsilon, fa, dr, Nr, nNCUT)
        
        dvec1h = dvec1 + k2tot*dt/2.0
        nNvh, Uvh,  Avh, Pvh = dvec1h[:Nr], dvec1h[Nr:2*Nr], dvec1h[2*Nr:3*Nr], dvec1h[3*Nr:]
        Gh = solve_G_constraint_fast(rvals, nNvh, Uvh, Avh, Pvh, epsilon, fa)
        Fh = solve_F_constraint_fast(rvals, Gvh, nNvh, Uvh, Avh, Pvh, epsilon, fa)
        k3tot = dMatterdt_for_RK4(rvals, Fh, Gh, dvec1 + k2tot*dt/2, epsilon, fa, dr, Nr, nNCUT)

        dvec1h = dvec1 + k3tot*dt
        nNvh, Uvh,  Avh, Pvh = dvec1h[:Nr], dvec1h[Nr:2*Nr], dvec1h[2*Nr:3*Nr], dvec1h[3*Nr:]
        Gh = solve_G_constraint_fast(rvals, nNvh, Uvh, Avh, Pvh, epsilon, fa)
        Fh = solve_F_constraint_fast(rvals, Gvh, nNvh, Uvh, Avh, Pvh, epsilon, fa)
        k4tot = dMatterdt_for_RK4(rvals, Fh, Gh, dvec1 + k3tot*dt, epsilon, fa, dr, Nr, nNCUT)

    else:
        k2tot = dMatterdt_for_RK4(rvals, F0, G0, dvec1 + k1tot*dt/2, epsilon, fa, dr, Nr, nNCUT)
        k3tot = dMatterdt_for_RK4(rvals, F0, G0, dvec1 + k2tot*dt/2, epsilon, fa, dr, Nr, nNCUT)
        k4tot = dMatterdt_for_RK4(rvals, F0, G0, dvec1 + k3tot*dt, epsilon, fa, dr, Nr, nNCUT)
    
    dvec2 = dvec1 + (1/6)*k1tot*dt + (1/3)*k2tot*dt + (1/3)*k3tot*dt + (1/6)*k4tot*dt
    
    # make sure functions are all flat across the origin
    
    #ans2 = np.zeros(np.shape(rvals))
    #num2 = nN0
    #den2 = nN0
    #np.divide(num2.astype(np.float64), den2.astype(np.float64), out=ans2, where=(num2 > nNCUT))
    
    dvec2[:Nr][dvec2[:Nr] < 0.0] = 0.0 
    #dvec2[Nr:2*Nr][dvec2[Nr:2*Nr] > 0.99] = 0.99
    #dvec2[Nr:2*Nr][dvec2[:Nr] < nsatinGeV3*1e-10] = -1e-10

    dvec2[:Nr] = enforce_rc_BC(dvec2[:Nr])
    dvec2[Nr:2*Nr] = enforce_rc_BC(dvec2[Nr:2*Nr], ZEROATRC=1)
    dvec2[2*Nr:3*Nr] = enforce_rc_BC(dvec2[2*Nr:3*Nr])
    dvec2[3*Nr:4*Nr] = enforce_rc_BC(dvec2[3*Nr:])

    return dvec2.astype(np.float64)
