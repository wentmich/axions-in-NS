(* GLOBALS *)
Msun = 1.989*10^30;(* kg *)
GN = 6.67*10^-11; (* SI *)
G =(*6.70883*10^(-57)*)1; (* eV^-2 *)
clight = 2.99792*10^8; (* m/s *)
eV2joules = 1.60218*10^(-19);
GeV2joules = 1.60218*10^(-10);
mN = 0.939563; (* GeV *)
hbarGeV = 6.582119569*10^(-25); (* GeV S *)
hbarJ = 1.05457*10^(-34); (* J s *)
hbarGeo = hbarJ * (GN / clight^4) * clight; (* m^2 *)
(*GNeV = 6.70883*10^(-39);*)
\[Sigma]N = 0.059;(* GeV *)(* eV*s *)
rkm = 10^3 / (hbarGeV*clight);
rm = 1/(hbarGeV*clight);

MeVperfm3\[LetterSpace]2\[LetterSpace]Jperm3 = (10^-3*
     GeV2joules) * (10^(-15))^(-3);
Jperm3\[LetterSpace]2\[LetterSpace]m2 = GN / clight^4;
GeV2m = GeV2joules*GN/clight^4;

RescaleMRPair\[LetterSpace]km\[LetterSpace]Msun[MRPair_, 
  scales_] := {(MRPair[[1]])*
   scales[[1]]/10^3, ((MRPair[[2]])*scales[[3]]*clight^2/GN)/Msun}
RescaleMR\[CapitalLambda]Triplet\[LetterSpace]km\[LetterSpace]Msun\
\[LetterSpace]dim0[MR\[CapitalLambda]Triplet_, 
  scales_] := {(MR\[CapitalLambda]Triplet[[1]])*
   scales[[1]]/
    10^3, ((MR\[CapitalLambda]Triplet[[2]])*scales[[3]]*clight^2/GN)/
   Msun, MR\[CapitalLambda]Triplet[[3]]}
RescaleMRPair\[LetterSpace]m\[LetterSpace]Msun[MRPair_, 
  scales_] := {(MRPair[[1]])*
   scales[[1]], ((MRPair[[2]])*scales[[3]]*clight^2/GN)/Msun}
RescaleMRPair\[LetterSpace]m\[LetterSpace]kg[MRPair_, 
  scales_] := {(MRPair[[1]])*
   scales[[1]], ((MRPair[[2]]*clight^2/GN)*scales[[1]]^3 * 
    scales[[2]])}
RescaleM\[LetterSpace]kg[mass_, 
  scales_] := ((mass*clight^2/GN)*scales[[1]]^3 * scales[[2]])
RescaleR\[LetterSpace]m[radius_, scales_] := (radius)*scales[[1]]

TOV\[LetterSpace]replacements = {\[Lambda]'[r] -> 
    D[Log[1/(1 - 2*M[r]/r)], r], \[Lambda]''[r] -> 
    D[-Exp[\[Lambda][r]]*(2*M[r]/r^2 - 8*Pi*r*\[Rho][r]), r],
   Exp[\[Lambda][r]] -> 1/(1 - 2*M[r]/r),
   Exp[-\[Lambda][r]] -> (1 - 2*M[r]/r),
   \[Lambda][r] -> Log[1/(1 - 2*M[r]/r)], 
   Derivative[1][p][
     r] -> -(((M[r] + 4*Pi*r^3*p[r])*(p[r] + \[Rho][r]))/(r*(r - 
           2*M[r]))),
   Derivative[2][p][r] -> 
    Simplify[
     D[-(((M[r] + 4*Pi*r^3*p[r])*(p[r] + \[Rho][r]))/(r*(r - 
             2*M[r]))), r]],
   Derivative[1][\[Nu]][
     r] -> -2*(Derivative[1][p][r])/(p[r] + \[Rho][r]),
   M'[r] -> 4*Pi*r^2*\[Rho][r]};




(* FUNCTIONS *)
(* SOLVE FOR MASS FUNCTION, DENSITY FUNCTION, MASS, AND RADIUS GIVEN \
EQUATION OF STATE AND CENTRAL DENSITY *)
GetGenericMassRadius[\[Rho]central_, EOS_, \[Rho]frac_, scales_] := 
 Module[{dMdr, dPdr, \[Rho]c = \[Rho]central, sols, Rstar, Mstar, 
   mass, density, rstart = 10.0/10^5, rend = 50.0, alpha, Acoef, 
   density\[LetterSpace]out, density\[LetterSpace]total, ddensitydr, 
   mass\[LetterSpace]out, mass\[LetterSpace]total, dEOSdrhoofr, 
   EOSscaled, dEOSscaleddrhoofr},
  (*EOSscaled[\[Rho]_]:=EOS[\[Rho]*scales[[2]]]/scales[[4]];*)
  dEOSdrhoofr[\[Rho]_] := Evaluate[D[EOS[\[Rho]], \[Rho]]];
  dMdr = 
   D[M[r], r] == 
    4*Pi*r^2*\[Rho][r]*(scales[[1]]^3* scales[[2]]/scales[[3]]);
  dPdr = 
   D[\[Rho][r], 
     r] == -((dEOSdrhoofr[scales[[2]]*\[Rho][r]])^(-1))*(M[r] + 
       4*Pi*r^3*EOS[\[Rho][r]*scales[[2]]]*
        scales[[1]]^3/scales[[3]])*(EOS[\[Rho][r]*scales[[2]]]/
         scales[[2]] + \[Rho][
         r])/(r*(r*scales[[1]]/scales[[3]] - 2*M[r]));
  sols = 
   NDSolve[{dMdr, dPdr, 
      M[rstart] == (4/3)*Pi*\[Rho]c*rstart^3*scales[[2]]*
        scales[[1]]^3/scales[[3]], \[Rho][
        rstart] == \[Rho]c - (2*Pi/3)*(\[Rho]c + 
           EOS[\[Rho]c*scales[[2]]]/scales[[2]])*(\[Rho]c*
            scales[[2]] + 3*EOS[\[Rho]c*scales[[2]]])*rstart^2*
         scales[[1]]^2, 
      WhenEvent[\[Rho][r] == \[Rho]c*\[Rho]frac, 
       "StopIntegration"]}, {M, \[Rho]}, {r, rstart, rend}, 
     PrecisionGoal -> 10, AccuracyGoal -> 10][[1]];
  mass[r_] := Evaluate[M[r] /. sols];
  density[r_] := Evaluate[\[Rho][r] /. sols];
  ddensitydr[r_] := Evaluate[D[density[r], r]];
  Rstar = Re[density["Domain"][[1]][[2]]];
  Mstar = Re[mass[Rstar]];
  (*alpha=-(ddensitydr[Rstar]/density[Rstar]);
  Acoef=density[Rstar];
  density\[LetterSpace]out[r_]:=Acoef*Exp[-alpha*(r - Rstar)];
  density\[LetterSpace]total[r_]:=If[r<=Rstar,density[r],
  density\[LetterSpace]out[r]];
  mass\[LetterSpace]out[r_]:=Mstar -((Acoef*(-1+Exp[alpha*(-r+
  Rstar)]))/alpha);
  mass\[LetterSpace]total[r_]:=If[r<=Rstar,mass[r],
  mass\[LetterSpace]out[r]];*)
  {{Rstar, Mstar}, {mass\[LetterSpace]total, 
    density\[LetterSpace]total}}]

(* SOLVE FOR MASS FUNCTION, DENSITY FUNCTION, MASS, RADIUS, AND TIDAL \
DEFORMABILITY GIVEN EQUATION OF STATE AND CENTRAL DENSITY *)
GetMRand\[CapitalLambda]NoAxions[\[Rho]central_, EOS_, \[Rho]frac_, 
  scales_] := 
 Module[{dMdr, dPdr, \[Rho]c = \[Rho]central, sols, Rstar, Mstar, 
   mass, density, rstart = 10^-5/scales[[1]], 
   rend = 5000.0*10^4/scales[[1]], alpha, Acoef, 
   density\[LetterSpace]out, density\[LetterSpace]total, ddensitydr, 
   mass\[LetterSpace]out, mass\[LetterSpace]total, dEOSdrhoofr, 
   EOSscaled, dEOSscaleddrhoofr, yequation, yinit, ysol, pressure, yR,
    Co, tidal\[LetterSpace]deformability},
  (*EOSscaled[\[Rho]_]:=EOS[\[Rho]*scales[[2]]]/scales[[4]];*)
  dEOSdrhoofr[\[Rho]_] := Evaluate[D[EOS[\[Rho]], \[Rho]]];
  dMdr = 
   D[M[r], r] == 
    4*Pi*r^2*\[Rho][r]*(scales[[1]]^3* scales[[2]]/scales[[3]]);
  dPdr = 
   D[\[Rho][r], 
     r] == -((dEOSdrhoofr[scales[[2]]*\[Rho][r]])^(-1))*(M[r] + 
       4*Pi*r^3*EOS[\[Rho][r]*scales[[2]]]*
        scales[[1]]^3/scales[[3]])*(EOS[\[Rho][r]*scales[[2]]]/
         scales[[2]] + \[Rho][
         r])/(r*(r*scales[[1]]/scales[[3]] - 2*M[r]));
  sols = 
   NDSolve[{dMdr, dPdr, 
      M[rstart] == (4/3)*Pi*\[Rho]c*rstart^3*scales[[2]]*
        scales[[1]]^3/scales[[3]], \[Rho][rstart] == \[Rho]c, 
      WhenEvent[\[Rho][r] == \[Rho]c*\[Rho]frac, 
       "StopIntegration"]}, {M, \[Rho]}, {r, rstart, 
      rend},(*Method->{"StiffnessSwitching",
     Method->{"ExplicitRungeKutta",Automatic}},AccuracyGoal->5,
     PrecisionGoal->4*)PrecisionGoal -> 2][[1]];
  mass[r_] := Evaluate[M[r] /. sols];
  density[r_] := Evaluate[\[Rho][r] /. sols];
  ddensitydr[r_] := Evaluate[D[density[r], r]];
  pressure[r_] := Evaluate[EOS[density[r]*scales[[2]]]];
  Rstar = Re[density["Domain"][[1]][[2]]];
  Mstar = Re[mass[Rstar]];
  
  yequation = (2*(3*r^2*scales[[1]]^2 + 
         2*(mass[r]^2*scales[[3]]^2 + 
            Pi*r^4*scales[[1]]^4*(pressure[
                 r]*(-9 + 16*Pi*r^2*scales[[1]]^2*pressure[r]) - 
               5*density[r]*scales[[2]]) + 
            r*scales[[1]]*mass[r]*
             scales[[3]]*(-3 + 
               2*Pi*r^2*
                scales[[1]]^2*(13*pressure[r] + 
                  5*density[r]*scales[[2]])) + (Pi*r^4*
               scales[[1]]^4*(r*scales[[1]] - 
                  2*mass[r]*scales[[3]])^2*ddensitydr[r]*
               scales[[2]]/scales[[1]])/(mass[r]*scales[[3]] + 
               4*Pi*r^3*scales[[1]]^3*pressure[r]))))/(r*
       scales[[1]]*(r*scales[[1]] - 2*mass[r]*scales[[3]])) == ((2*
         y[r]*(r*scales[[1]] - mass[r]*scales[[3]] + 
           2*Pi*r^3*
            scales[[1]]^3*(pressure[r] - density[r]*scales[[2]])) + 
        r*scales[[1]]*(r*scales[[1]] - 
           2*mass[r]*scales[[3]])*(y[r]^2/(Rstar*scales[[1]]) + 
           Derivative[1][y][r]/scales[[1]])))/(Rstar*scales[[1]]);
  yinit = y[rstart] == Rstar/rstart;
  sols = 
   NDSolve[{yequation, yinit}, {y}, {r, rstart, Rstar}, 
     PrecisionGoal -> 6][[1]];
  ysol[r_] := Evaluate[y[r] /. sols];
  
  yR = ysol[Rstar];
  (*Print[yR];*)
  Co = Mstar*scales[[3]]/(Rstar*scales[[1]]);
  tidal\[LetterSpace]deformability = (16*(1 - 2*Co)^2*(2 + 
        2*Co*(-1 + yR) - yR))/(30*
       Co*(6 - 3*yR + 
         Co*(3*(-8 + 5*yR) + 
            2*Co*(13 - 11*yR + Co*(-2 + 3*yR + 2*Co*(1 + yR))))) + 
      45*(1 - 2*Co)^2*(2 + 2*Co*(-1 + yR) - yR)*Log[1 - 2*Co]);
  
  alpha = -(ddensitydr[Rstar]/density[Rstar]);
  Acoef = density[Rstar];
  density\[LetterSpace]out[r_] := Acoef*Exp[-alpha*(r - Rstar)];
  density\[LetterSpace]total[r_] := 
   If[r <= Rstar, density[r], density\[LetterSpace]out[r]];
  mass\[LetterSpace]out[r_] := 
   Mstar - ((Acoef*(-1 + Exp[alpha*(-r + Rstar)]))/alpha);
  mass\[LetterSpace]total[r_] := 
   If[r <= Rstar, mass[r], mass\[LetterSpace]out[r]];
  {{Rstar, Mstar, 
    tidal\[LetterSpace]deformability}, {mass\[LetterSpace]total, 
    density\[LetterSpace]total}}]

(* SOLVE FOR AXION AND AXION DERIVATIVE FIELDS *)
GetAxionField[massradiussol_, mass_, density_, ma_, fa_, scales_] := 
 Module[{density\[LetterSpace]coefficient\[LetterSpace]sub, 
   d\[Theta]pdr\[LetterSpace]rscaled, 
   d\[Theta]dr\[LetterSpace]rscaled, mythetaTfncsols, NSradiusscaled, 
   r, rt, \[Theta]sol, \[Theta]primesol, rstart = 10^-5/scales[[1]], 
   NSradius = 
    massradiussol[[1]], \[Theta]solin, \[Theta]solout, \
\[Theta]primesolin, \[Theta]primesolout},
  d\[Theta]pdr\[LetterSpace]rscaled = 
   D[\[Theta]Tp[rt], 
     rt] == (ma^2*
       scales[[1]]^2/hbarGeo^2)*(1 - 
        2*mass[rt]*scales[[3]]/(rt*scales[[1]]))^(-1) * (1 - 
       density[rt]*scales[[2]]*hbarGeo^3*\[Sigma]N/(4*fa^2*ma^2*mN))*
     Sin[\[Theta]T[rt]]*(Cos[\[Theta]T[rt]]/Abs[Cos[\[Theta]T[rt]]]);
  d\[Theta]dr\[LetterSpace]rscaled = 
   D[\[Theta]T[rt], rt] == \[Theta]Tp[rt];
  mythetaTfncsols = 
   NDSolve[{d\[Theta]pdr\[LetterSpace]rscaled, 
      d\[Theta]dr\[LetterSpace]rscaled, \[Theta]T[NSradius] == 
       2*Pi*Exp[-ma*scales[[1]]*NSradius/hbarGeo], \[Theta]Tp[
        NSradius] == -2*Pi*
        Exp[-ma*scales[[1]]*NSradius/hbarGeo]*(1/NSradius + 
          scales[[1]]*ma/hbarGeo)}, {\[Theta]T, \[Theta]Tp}, {rt, 
      rstart, NSradius}, PrecisionGoal -> 4][[1]];
  \[Theta]solin[rt_] := Evaluate[\[Theta]T[rt] /. mythetaTfncsols];
  \[Theta]primesolin[rt_] := 
   Evaluate[\[Theta]Tp[rt] /. mythetaTfncsols];
  \[Theta]solout[rt_] := (2*Pi*NSradius/rt)*
    Exp[-ma*scales[[1]]*rt/hbarGeo];
  \[Theta]primesolout[
    rt_] := (-2*Pi*
      NSradius/rt)*(Exp[-ma*scales[[1]]*rt/hbarGeo]/
       rt + (scales[[1]]*ma/hbarGeo)*
       Exp[-ma*scales[[1]]*rt/hbarGeo]);
  \[Theta]sol[rt_] := 
   If[rt < NSradius, \[Theta]solin[rt], \[Theta]solout[rt]];
  \[Theta]primesol[rt_] := 
   If[rt < NSradius, \[Theta]primesolin[rt], \[Theta]primesolout[
     rt]];
  {{\[Theta]sol, \[Theta]primesol}}]

(* SOLVE FOR DYNAMIC AXION AND AXION DERIVATIVE FIELDS *)
GetDynamicAxionField[massradiussol_, mass_, density_, ma_, fa_, 
  scales_] := 
 Module[{density\[LetterSpace]coefficient\[LetterSpace]sub, 
   d\[Theta]pdr\[LetterSpace]rscaled, 
   d\[Theta]dr\[LetterSpace]rscaled, mythetaTfncsols, NSradiusscaled, 
   r, rt, \[Theta]sol, \[Theta]primesol, rstart = 10^-5/scales[[1]], 
   NSradius = 
    massradiussol[[1]], \[Theta]solin, \[Theta]solout, \
\[Theta]primesolin, \[Theta]primesolout, 
   d2\[Theta]dr2\[LetterSpace]rscaled},
  d2\[Theta]dr2\[LetterSpace]rscaled = (1 - 
       2*mass[rt]*scales[[3]]/(rt*scales[[1]]))* 
     D[\[Theta]T[rt], rt, 
      rt] == (ma^2*scales[[1]]^2/hbarGeo^2)* (1 - 
       density[rt]*scales[[2]]*hbarGeo^3*\[Sigma]N/(4*fa^2*ma^2*mN))*
     Sin[\[Theta]T[rt]]*(Cos[\[Theta]T[rt]]/Abs[Cos[\[Theta]T[rt]]]);
  mythetaTfncsols = 
   NDSolve[{d\[Theta]pdr\[LetterSpace]rscaled, 
      d\[Theta]dr\[LetterSpace]rscaled, \[Theta]T[NSradius] == 
       2*Pi*Exp[-ma*scales[[1]]*NSradius/hbarGeo], \[Theta]Tp[
        NSradius] == -2*Pi*
        Exp[-ma*scales[[1]]*NSradius/hbarGeo]*(1/NSradius + 
          scales[[1]]*ma/hbarGeo)}, {\[Theta]T, \[Theta]Tp}, {rt, 
      rstart, NSradius}, PrecisionGoal -> 4][[1]];
  \[Theta]solin[rt_] := Evaluate[\[Theta]T[rt] /. mythetaTfncsols];
  \[Theta]primesolin[rt_] := 
   Evaluate[\[Theta]Tp[rt] /. mythetaTfncsols];
  \[Theta]solout[rt_] := (2*Pi*NSradius/rt)*
    Exp[-ma*scales[[1]]*rt/hbarGeo];
  \[Theta]primesolout[
    rt_] := (-2*Pi*
      NSradius/rt)*(Exp[-ma*scales[[1]]*rt/hbarGeo]/
       rt + (scales[[1]]*ma/hbarGeo)*
       Exp[-ma*scales[[1]]*rt/hbarGeo]);
  \[Theta]sol[rt_] := 
   If[rt < NSradius, \[Theta]solin[rt], \[Theta]solout[rt]];
  \[Theta]primesol[rt_] := 
   If[rt < NSradius, \[Theta]primesolin[rt], \[Theta]primesolout[
     rt]];
  {{\[Theta]sol, \[Theta]primesol}}]

(* SOLVE FOR AXION DENSITY AND PRESSURE *)
Clear[GetAxionEnergyPressureAndMass]
GetAxionEnergyPressureAndMass[\[Theta]sol_, \[Theta]primesol_, 
  massradiussol_, mass_, density_, ma_, fa_, scales_] := 
 Module[{vascaled, \[Rho]asol, pasol, rstart = 10^-5/scales[[1]], 
   Masol, NSradius = massradiussol[[1]], matotal, madiffeq, mainside, 
   masols, rt, mainsol},
  vascaled[r_] := 
   Evaluate[-(4*fa^2*ma^2/hbarGeo^3 - \[Sigma]N*density[r]*
         scales[[2]]/mN)*RealAbs[Cos[\[Theta]sol[r]]]];
  \[Rho]asol[r_] := 
   Evaluate[(1/
       2)*((1 - 
          2*scales[[3]]*mass[r]/(scales[[1]]*r))*(2*
           fa/scales[[1]])^2*(1/hbarGeo)*\[Theta]primesol[r]^2 + 
       2*vascaled[r])];
  pasol[r_] := 
   Evaluate[(1/
       2)*(-(1 - 2*scales[[3]]*mass[r]/(scales[[1]]*r))*(2*
           fa/scales[[1]])^2*(1/hbarGeo)*\[Theta]primesol[r]^2 + 
       2*vascaled[r])];
  matotal = 
   NIntegrate[\[Rho]asol[rt]*4*Pi*scales[[1]]^3*rt^2/scales[[3]], {rt,
      rstart, NSradius}];
  madiffeq = 
   D[mainside[rt], rt]*scales[[3]]/scales[[1]] == 
    4*Pi*rt^2*scales[[1]]^2*\[Rho]asol[rt];
  masols = 
   NDSolve[{madiffeq, mainside[NSradius] == matotal}, {mainside}, {rt,
       rstart, NSradius}, PrecisionGoal -> 4][[1]];
  mainsol[rt_] := Evaluate[mainside[rt] /. masols];
  Masol[r_] := If[r < NSradius, mainsol[r], matotal];
  {{\[Rho]asol, pasol, Masol}}]
(* I should define a differential equation for the Ma(r) function so \
that it gets an interpolating function. Quicker evaluation. *)

(* SOLVE FOR AXION METRIC PERTURBATIONS *)
GetDeltaAndZetaMetricPerturbations[massradiussol_, mass_, density_, 
  pressure_, axion\[LetterSpace]density_, 
  axion\[LetterSpace]pressure_, Matotal_, scales_] := 
 Module[{d\[CapitalDelta]dr, 
   d\[Zeta]dr, \[CapitalDelta]initial, \[Zeta]initial, 
   NSradius = massradiussol[[1]], NSmass = massradiussol[[2]], 
   rstart = 
    10^-5/scales[[1]], \[CapitalDelta]and\[Zeta]fncsols, \
\[CapitalDelta]solin, \[Zeta]solin, \[CapitalDelta]solout, \
\[Zeta]solout, \[CapitalDelta]sol, \[Zeta]sol, rt, rtt},
  d\[CapitalDelta]dr = 
   D[\[CapitalDelta][rtt], 
     rtt] == \[CapitalDelta][rtt]*
      scales[[1]]*(8*Pi*scales[[1]]^2*scales[[2]]*rtt^2*
          density[rtt] - 1)/(scales[[1]]*rtt - 
         2*scales[[3]]*mass[rtt]) + (8*Pi*scales[[1]]^3*rtt^2*
        axion\[LetterSpace]density[rtt])/(scales[[1]]*rtt - 
        2*scales[[3]]*mass[rtt]);
  d\[Zeta]dr = 
   D[\[Zeta][rtt], 
     rtt] == \[CapitalDelta][rtt]*
      scales[[1]]*(8*Pi*scales[[1]]^2*rtt^2*pressure[rtt] + 
         1)/(scales[[1]]*rtt - 2*scales[[3]]*mass[rtt]) + (8*Pi*
        scales[[1]]^3*rtt^2*
        axion\[LetterSpace]pressure[rtt])/(scales[[1]]*rtt - 
        2*scales[[3]]*mass[rtt]);
  \[CapitalDelta]initial = \[CapitalDelta][NSradius] == -2*Matotal*
     scales[[3]]/(2*NSmass*scales[[3]] - NSradius*scales[[1]]);
  \[Zeta]initial = \[Zeta][NSradius] == 
    2*Matotal*
     scales[[3]]/(2*NSmass*scales[[3]] - NSradius*scales[[1]]);
  \[CapitalDelta]and\[Zeta]fncsols = 
   NDSolve[{d\[CapitalDelta]dr, 
      d\[Zeta]dr, \[CapitalDelta]initial, \[Zeta]initial}, {\
\[CapitalDelta], \[Zeta]}, {rtt, rstart, NSradius}, 
     PrecisionGoal -> 4][[1]];
  \[CapitalDelta]solin[rt_] := 
   Evaluate[\[CapitalDelta][rt] /. \[CapitalDelta]and\[Zeta]fncsols];
  \[Zeta]solin[rt_] := 
   Evaluate[\[Zeta][rt] /. \[CapitalDelta]and\[Zeta]fncsols];
  \[CapitalDelta]solout[rt_] := 
   Evaluate[-2*Matotal*
     scales[[3]]/(2*NSmass*scales[[3]] - rt*scales[[1]])];
  \[Zeta]solout[rt_] := 
   Evaluate[
    2*Matotal*scales[[3]]/(2*NSmass*scales[[3]] - rt*scales[[1]])];
  \[CapitalDelta]sol[rt_] := 
   Evaluate[
    If[rt < NSradius, \[CapitalDelta]solin[rt], 
     Evaluate[\[CapitalDelta]solout[rt]]]];
  \[Zeta]sol[rt_] := 
   Evaluate[
    If[rt < NSradius, \[Zeta]solin[rt], 
     Evaluate[\[Zeta]solout[rt]]]];
  {{\[Zeta]sol, \[CapitalDelta]sol}}]

(* SOLVE FOR AXION METRIC PERTURBATIONS *)
Clear[GetTidalDeformabilityWithAxions]
GetTidalDeformabilityWithAxions[massradiussol_, M_, p_, \[Rho]_, 
  axionmassfnc_, pa_, \[Rho]a_, \[CapitalDelta]_, \[Zeta]_, scales_] :=
  Module[{yequqation, yinitial, yfncsols, yfncsol, 
   Rs = massradiussol[[1]], rstart = 10^-5/scales[[1]], 
   tidal\[LetterSpace]deformability, yatsfc, Co, sourceterm, yeq2, 
   Ma = axionmassfnc[massradiussol[[1]]], Mns = massradiussol[[2]], 
   tidal\[LetterSpace]deformability\[LetterSpace]noaxions, 
   mscale = scales[[3]], 
   rscale = scales[[1]], \[Rho]scale = scales[[2]], 
   tidal\[LetterSpace]deformability\[LetterSpace]axions, y0atsfc, 
   y1atsfc, y0, y1, y0eq, y1eq, sourceterm0, sourceterm1, y0initial, 
   y0fncsols, y0fncsol, y1initial, y1fncsols, y1fncsol},
  y0eq = 
   y0'[r]/rscale == InputForm[
    r^(-2) Rs^(-1) 
     rscale^(-3) (r rscale - 2 mscale M[r])^(-2) (mscale M[r] + 
       4 Pi r^3 rscale^3 p[r])^(-1) (
      4 mscale^3 
       M[r]^3 (Rs^2 rscale^2 - r rscale 
        y0[r] (Rs rscale + r rscale y0[r])) + 
      4 Pi r^5 
       rscale^5 (
        p[r] (6 Rs^2 rscale^2 + 
          r rscale (64 Pi^2 r^3 Rs^2 rscale^5 p[r]^2 - 
            y0[r] (2 Rs rscale + r rscale y0[r]) - 4 Pi r Rs rscale^2 
            p[r] (9 Rs rscale + r rscale y0[r]) + 
            4 Pi r Rs 
             rscale^2 \[Rho]scale ((-5) Rs rscale + 
              r rscale y0[r]) \[Rho][r])) + 
        r Rs^2 rscale^2 \[Rho]scale Derivative[1][\[Rho]][r]) + 
      2 mscale^2 r rscale 
       M[r]^2 ((-6) Rs^2 rscale^2 + 
        r rscale (2 r rscale y0[r]^2 + 
          4 Pi r rscale 
           p[r] (5 Rs rscale - 2 r rscale y0[r]) (3 Rs rscale + 
            r rscale y0[r]) + 
          Rs rscale 
           y0[r] (3 - 4 Pi r^2 rscale^2 \[Rho]scale \[Rho][r]) + 
          4 Pi r Rs^2 
           rscale^3 (5 \[Rho]scale \[Rho][r] + 
            2 r \[Rho]scale Derivative[1][\[Rho]][r]))) + 
      mscale r^2 rscale^2 
       M[r] (6 Rs^2 rscale^2 + 
        r rscale ((-r) rscale y0[r]^2 + 
          32 Pi^2 r^3 Rs rscale^4 
           p[r]^2 (15 Rs rscale + r rscale y0[r]) + 
          2 Rs rscale 
           y0[r] (-1 + 2 Pi r^2 rscale^2 \[Rho]scale \[Rho][r]) + 
          4 Pi r rscale 
           p[r] ((-21) Rs^2 rscale^2 + 
            r rscale (y0[r] (5 Rs rscale + 4 r rscale y0[r]) - 8 Pi r 
              Rs rscale^2 \[Rho]scale ((-5) Rs rscale + 
               r rscale y0[r]) \[Rho][r])) - 4 Pi r Rs^2 
          rscale^3 (5 \[Rho]scale \[Rho][r] + 
           4 r \[Rho]scale Derivative[1][\[Rho]][r]))))];
  (* this following expression is what nico has in his paper *)
  (*y0eq=y0'[r]/rscale==-(y0[r]^2/(Rs*rscale)) + (2*Rs*((y0[r]*(-(r*
  rscale) - mscale*M[r] + (2*Pi*r^4*rscale^4*(-p[
  r] + \[Rho]scale*\[Rho][r]))/(r*rscale - 2*mscale*M[r])))/(Rs*
  rscale) + (3*r^2*rscale^2 + 2*(mscale^2*M[r]^2 + Pi*r^4*rscale^4*(p[
  r]*(-9 + 16*Pi*r^2*rscale^2*p[r]) - 5*\[Rho]scale*\[Rho][r]) + 
  mscale*r*rscale*M[r]*(-3 + 2*Pi*r^2*rscale^2*(13*p[r] + 
  5*\[Rho]scale*\[Rho][r]))))/(r*rscale - 2*mscale*M[r])^2 + (2*Pi*
  r^4*rscale^3*\[Rho]scale*Derivative[1][\[Rho]][r])/(mscale*M[r] + 4*
  Pi*r^3*rscale^3*p[r])))/(r^2*rscale);*)
  sourceterm0[rt_] := Evaluate[y0eq /. {r -> rt}];
  (*Print[sourceterm[Rs/2]];*)
  y0initial = y0[rstart] == Rs/rstart;
  y0fncsols = 
   NDSolve[{sourceterm0[rt], y0initial}, {y0}, {rt, rstart, Rs}, 
     PrecisionGoal -> 4, AccuracyGoal -> 10][[1]];
  y0fncsol[rt_] := Evaluate[y0[rt] /. y0fncsols];
  y0atsfc = y0fncsol[Rs];
  y1eq = 
   y1'[r]/rscale == InputForm[
    r^(-2) Rs^(-1) 
     rscale^(-3) (r rscale - 2 mscale M[r])^(-2) (mscale M[r] + 
       4 Pi r^3 rscale^3 p[r])^(-1) (
      4 Pi r^3 Rs rscale^4 
       pa[r] (mscale^2 
         M[r]^2 ((-6) Rs rscale + 2 r rscale y0fncsol[r]) + 
        mscale r rscale 
         M[r] (9 Rs rscale - 5 r rscale y0fncsol[r] - 24 Pi r^2 
          rscale^2 p[r] ((-Rs) rscale + r rscale y0fncsol[r])) + 
        2 r^2 rscale^2 ((-Rs) rscale + 32 Pi^2 r^4 Rs rscale^5 p[r]^2 + 
          r rscale y0fncsol[r] + 
          2 Pi r^2 rscale^2 
           p[r] (Rs rscale + 3 r rscale y0fncsol[r]))) - 2 mscale^3 
      M[r]^3 (2 r rscale (Rs rscale + 2 r rscale y0fncsol[r]) y1[r] - 
       Rs rscale (Rs rscale - r rscale 
        y0fncsol[r]) (\[CapitalDelta][r] - \[Zeta][r])) + 
      4 Pi r^5 
       rscale^5 (
        32 Pi^2 r^4 Rs^2 rscale^6 
         p[r]^3 (\[CapitalDelta][r] + \[Zeta][r]) + 
        4 Pi r^2 Rs rscale^3 
         p[r]^2 ((-r) rscale y1[r] - 5 Rs rscale \[Zeta][r] - r 
          rscale y0fncsol[r] \[Zeta][
           r] + \[CapitalDelta][r] (Rs rscale + 
            4 r rscale y0fncsol[r] + 
            16 Pi r^2 Rs rscale^3 \[Rho]scale \[Rho][r]) + 
          16 Pi r^2 Rs rscale^3 \[Rho]a[r]) + 
        p[r] (2 r rscale 
           y1[r] ((-r) rscale y0fncsol[r] + 
            Rs rscale (-1 + 2 Pi r^2 rscale^2 \[Rho]scale \[Rho][r])) + 
          Rs rscale (
            2 Rs rscale \[Zeta][r] (1 - 6 Pi r^2 
              rscale^2 \[Rho]scale \[Rho][r]) + \[CapitalDelta][
              r] ((-4) Rs 
               rscale (-1 + Pi r^2 rscale^2 \[Rho]scale \[Rho][r]) + 
              y0fncsol[r] (r rscale + 
                4 Pi r^3 rscale^3 \[Rho]scale \[Rho][r])) + 
            4 Pi r^2 
             rscale^2 (Rs rscale + r rscale y0fncsol[r]) \[Rho]a[r])) + 
        Rs rscale ((-2) (Rs rscale - r rscale y0fncsol[r]) \[Rho]a[r] + 
          r Rs rscale^2 (
            rscale^(-1) \[Rho]scale \[CapitalDelta][r] 
             Derivative[1][\[Rho]][r] + 
            rscale^(-1) \[Rho]scale \[Zeta][r] 
             Derivative[1][\[Rho]][r] + 
            rscale^(-1) Derivative[1][\[Rho]a][r]))) - mscale^2 r 
      rscale M[r]^2 (
       2 r rscale 
        y1[r] ((-4) r rscale y0fncsol[r] + 
         4 Pi r^2 rscale^2 p[r] (Rs rscale + 4 r rscale y0fncsol[r]) + 
         Rs rscale (-3 + 4 Pi r^2 rscale^2 \[Rho]scale \[Rho][r])) + 
       Rs rscale (\[CapitalDelta][r] (
           y0fncsol[r] (r rscale + 40 Pi r^3 rscale^3 p[r] + 
             8 Pi r^3 rscale^3 \[Rho]scale \[Rho][r]) - 2 Rs 
           rscale (-5 + 12 Pi r^2 rscale^2 \[Rho]scale \[Rho][r] + 
            8 Pi r^3 rscale^2 \[Rho]scale 
             Derivative[1][\[Rho]][r])) + \[Zeta][r] (
           r rscale y0fncsol[r] - 16 Pi r^2 rscale^2 
           p[r] (3 Rs rscale + r rscale y0fncsol[r]) - 2 Rs 
           rscale (-1 + 12 Pi r^2 rscale^2 \[Rho]scale \[Rho][r] + 
            8 Pi r^3 rscale^2 \[Rho]scale Derivative[1][\[Rho]][r])) - 
         8 Pi r^2 
         rscale^2 ((-3) (Rs rscale - r rscale y0fncsol[r]) \[Rho]a[r] + 
          2 r Rs rscale Derivative[1][\[Rho]a][r]))) + 
      mscale r^2 rscale^2 
       M[r] (2 r rscale 
         y1[r] (16 Pi^2 r^4 Rs rscale^5 p[r]^2 - r rscale y0fncsol[r] + 
          Rs rscale (-1 + 2 Pi r^2 rscale^2 \[Rho]scale \[Rho][r]) - 
          2 Pi r^2 rscale^2 
          p[r] ((-8) r rscale y0fncsol[r] + 
           Rs rscale (-5 + 8 Pi r^2 rscale^2 \[Rho]scale \[Rho][r]))) + 
        Rs rscale (
          2 \[Zeta][r] (Rs rscale + 
            16 Pi^2 r^4 rscale^4 
             p[r]^2 (8 Rs rscale + r rscale y0fncsol[r]) - 6 Pi r^2 
            Rs rscale^3 \[Rho]scale \[Rho][r] + 
            2 Pi r^2 rscale^2 
             p[r] ((-2) r rscale y0fncsol[r] + 
              Rs rscale (-7 + 
                24 Pi r^2 rscale^2 \[Rho]scale \[Rho][r])) - 8 Pi r^3 
            Rs rscale^3 \[Rho]scale 
            Derivative[1][\[Rho]][r]) + \[CapitalDelta][
            r] ((-128) Pi^2 r^5 rscale^5 p[r]^2 y0fncsol[r] + 
            y0fncsol[r] (r rscale + 
              4 Pi r^3 rscale^3 \[Rho]scale \[Rho][r]) - 4 Pi r^2 
            rscale^2 
            p[r] (Rs 
              rscale (9 - 40 Pi r^2 rscale^2 \[Rho]scale \[Rho][r]) + 
             r rscale 
              y0fncsol[r] (-3 + 
               8 Pi r^2 rscale^2 \[Rho]scale \[Rho][r])) - 4 Rs 
            rscale (-1 + Pi r^2 rscale^2 \[Rho]scale \[Rho][r] + 
             4 Pi r^3 rscale^2 \[Rho]scale Derivative[1][\[Rho]][r])) - 
          4 Pi r^2 
          rscale^2 (((-9) Rs rscale + 7 r rscale y0fncsol[r] + 
             8 Pi r^2 rscale^2 
              p[r] ((-3) Rs rscale + r rscale y0fncsol[r])) \[Rho]a[
             r] + 4 r Rs rscale Derivative[1][\[Rho]a][r]))))];
  sourceterm1[rt_] := Evaluate[y1eq /. {r -> rt}];
  (*Print[sourceterm[Rs/2]];*)
  y1initial = y1[rstart] == 0;
  y1fncsols = 
   NDSolve[{sourceterm1[rt], y1initial}, {y1}, {rt, rstart, Rs}, 
     PrecisionGoal -> 4, AccuracyGoal -> 10][[1]];
  y1fncsol[rt_] := Evaluate[y1[rt] /. y1fncsols];
  y1atsfc = y1fncsol[Rs];
  (*Print[y0atsfc];*)
  tidal\[LetterSpace]deformability\[LetterSpace]noaxions = InputForm[
   Rational[8, 15] Mns^(-1) mscale^(-1) Rs^2 
    rscale^2 (2 Mns mscale - Rs rscale) (
     4 Mns mscale ((-3) Mns mscale Rs^6 rscale^6 (-2 + y0atsfc)^2 + 
       16 Mns^7 mscale^7 (-1 + y0atsfc^2) + 
       8 Mns^6 mscale^6 Rs rscale (5 - 4 y0atsfc + y0atsfc^2) + 
       3 Mns^2 mscale^2 Rs^5 rscale^5 (28 - 32 y0atsfc + 9 y0atsfc^2) - 
       4 Mns^5 mscale^5 Rs^2 
       rscale^2 (34 - 60 y0atsfc + 27 y0atsfc^2) + 
       2 Mns^4 mscale^4 Rs^3 
        rscale^3 (130 - 204 y0atsfc + 77 y0atsfc^2) - Mns^3 mscale^3 
       Rs^4 rscale^4 (220 - 292 y0atsfc + 94 y0atsfc^2)) + 
     6 Mns mscale Rs^2 
      rscale^2 (2 Mns mscale - Rs rscale)^3 (Rs rscale (-2 + y0atsfc) - 
        2 Mns mscale (-1 + y0atsfc))^2 
      Log[1 - 2 Mns mscale (Rs^(-1)/rscale)]) (
      2 Mns mscale (2 Mns^2 mscale^2 Rs^2 rscale^2 (13 - 11 y0atsfc) - 
        3 Rs^4 rscale^4 (-2 + y0atsfc) + 
        4 Mns^4 mscale^4 (1 + y0atsfc) + 
        2 Mns^3 mscale^3 Rs rscale (-2 + 3 y0atsfc) + 
        3 Mns mscale Rs^3 rscale^3 (-8 + 5 y0atsfc)) + 
      3 Rs^2 rscale^2 ((-2) Mns mscale + Rs rscale)^2 ((-Rs) 
         rscale (-2 + y0atsfc) + 2 Mns mscale (-1 + y0atsfc)) 
       Log[1 - 2 Mns mscale (Rs^(-1)/rscale)])^(-2)];
  tidal\[LetterSpace]deformability\[LetterSpace]axions = InputForm[
   Rational[8, 15] Mns^(-1) mscale^(-1) Rs^2 
    rscale^2 (2 Mns mscale - Rs 
     rscale) (
      2 Mns mscale (2 Mns^2 mscale^2 Rs^2 rscale^2 (13 - 11 y0atsfc) - 
        3 Rs^4 rscale^4 (-2 + y0atsfc) + 
        4 Mns^4 mscale^4 (1 + y0atsfc) + 
        2 Mns^3 mscale^3 Rs rscale (-2 + 3 y0atsfc) + 
        3 Mns mscale Rs^3 rscale^3 (-8 + 5 y0atsfc)) + 
      3 Rs^2 rscale^2 ((-2) Mns mscale + Rs rscale)^2 ((-Rs) 
         rscale (-2 + y0atsfc) + 2 Mns mscale (-1 + y0atsfc)) 
       Log[1 - 2 Mns mscale (Rs^(-1)/rscale)])^(-2) (
     4 Mns mscale (15 Ma mscale Rs^6 rscale^6 (-2 + y0atsfc)^2 - 3 
       Mns mscale Rs^5 
       rscale^5 (-2 + y0atsfc) (Rs rscale (-2 + y0atsfc) + 
        2 Ma mscale (-36 + 23 y0atsfc)) + 
       Mns^2 mscale^2 Rs^4 
        rscale^4 (3 Rs rscale (28 - 32 y0atsfc + 9 y0atsfc^2) + 
         2 Ma mscale (586 - 772 y0atsfc + 247 y0atsfc^2)) + 
       2 Mns^4 mscale^4 Rs^2 
        rscale^2 (Rs rscale (130 - 204 y0atsfc + 77 y0atsfc^2) + 
         Ma mscale (404 - 710 y0atsfc + 315 y0atsfc^2)) - Mns^3 
       mscale^3 Rs^3 
       rscale^3 (Rs rscale (220 - 292 y0atsfc + 94 y0atsfc^2) + 
        Ma mscale (1452 - 2252 y0atsfc + 841 y0atsfc^2)) + 
       8 Mns^6 mscale^6 (Ma mscale (9 + 2 y0atsfc - 11 y0atsfc^2) + 
         Rs rscale (5 - 4 y0atsfc + y0atsfc^2 - 4 y1atsfc)) - 4 Mns^5 
       mscale^5 Rs 
       rscale (Ma mscale (61 - 56 y0atsfc + 19 y0atsfc^2) + 
        Rs rscale (34 - 60 y0atsfc + 27 y0atsfc^2 - 2 y1atsfc)) + 
       16 Mns^7 mscale^7 (-1 + y0atsfc^2 + 2 y1atsfc)) - 3 Ma mscale 
     Rs^2 rscale^2 (2 Mns mscale - Rs rscale)^3 (
       Rs rscale (-2 + y0atsfc) - 2 Mns mscale (-1 + y0atsfc))^2 
     Log[1 - 2 Mns mscale (Rs^(-1)/rscale)]^2 - 2 Ma 
     mscale (2 Mns mscale - Rs 
      rscale) ((-3) Rs^6 rscale^6 (-2 + y0atsfc)^2 + 
      16 Mns^6 mscale^6 (-1 + y0atsfc^2) + 
      16 Mns^5 mscale^5 Rs rscale (2 - 2 y0atsfc + y0atsfc^2) + 
      6 Mns mscale Rs^5 rscale^5 (16 - 18 y0atsfc + 5 y0atsfc^2) - 6 
      Mns^2 mscale^2 Rs^4 rscale^4 (46 - 60 y0atsfc + 19 y0atsfc^2) + 
      8 Mns^3 mscale^3 Rs^3 rscale^3 (43 - 67 y0atsfc + 25 y0atsfc^2) - 
      4 Mns^4 mscale^4 Rs^2 rscale^2 (42 - 80 y0atsfc + 37 y0atsfc^2))
       Log[Rs rscale] + 
     2 Ma mscale (2 Mns mscale - Rs 
       rscale) ((-3) Rs^6 rscale^6 (-2 + y0atsfc)^2 + 
       16 Mns^6 mscale^6 (-1 + y0atsfc^2) + 
       16 Mns^5 mscale^5 Rs rscale (2 - 2 y0atsfc + y0atsfc^2) + 
       6 Mns mscale Rs^5 rscale^5 (16 - 18 y0atsfc + 5 y0atsfc^2) - 6 
       Mns^2 mscale^2 Rs^4 rscale^4 (46 - 60 y0atsfc + 19 y0atsfc^2) + 
       8 Mns^3 mscale^3 Rs^3 
        rscale^3 (43 - 67 y0atsfc + 25 y0atsfc^2) - 4 Mns^4 mscale^4 
       Rs^2 rscale^2 (42 - 80 y0atsfc + 37 y0atsfc^2)) 
      Log[(-2) Mns mscale + Rs rscale] + 
     6 Rs^2 rscale^2 (2 Mns mscale - Rs rscale)^3 (
        Rs rscale (-2 + y0atsfc) - 2 Mns mscale (-1 + y0atsfc))^2 
      Log[1 - 2 Mns mscale (Rs^(-1)/rscale)] ((-4) Ma mscale + 
       Mns mscale - Ma mscale Log[Rs rscale] + 
       Ma mscale Log[(-2) Mns mscale + Rs rscale]))];
  {{tidal\[LetterSpace]deformability\[LetterSpace]axions, 
    tidal\[LetterSpace]deformability\[LetterSpace]noaxions}}]





(* MR SOLUTION *)
FILEBASE = "eos_nb0.1_G0.04_nb_0.23_G_0.008";
EOSPATH = 
  "/Users/wentmich/Documents/uiuc/research/axions-in-NS/EOS-MR/input_\
stable_eos_files_p_of_nb_fixed/";
MRPATH = 
  "/Users/wentmich/Documents/uiuc/research/axions-in-NS/EOS-MR/output_\
stable_eos_files_p_of_nb_fixed/";
EOSData = Import[StringJoin[EOSPATH, FILEBASE, ".csv"], "CSV"];
MRData = 
  Import[StringJoin[MRPATH, FILEBASE, "_observables.dat"](*,"CSV"*)];
EOSDataLimited = EOSData[[1 ;; Dimensions[EOSData][[1]], 8 ;; 9]];
MRDataLimited = MRData[[1 ;; Dimensions[MRData][[1]], {2, 3}]];
M\[CapitalLambda]DataLimited = 
  MRData[[1 ;; Dimensions[MRData][[1]], {3, 5}]];
pressureEOS = 
  Interpolation[
   EOSDataLimited]; (* rho and p are in MeV / fm^3, so we need to \
rescale *) 
(* I actually shouldn't need to rescale. That should just be wrapped \
up in whatever I choose as my \[Rho]scale variable in the TOV solver. \
*)
dpressureEOS[\[Rho]_] := Evaluate[D[pressureEOS[\[Rho]], \[Rho]]];
maxdensity = Max[EOSDataLimited[[1 ;; Dimensions[EOSData][[1]], 1]]];
mindensity = Min[EOSDataLimited[[1 ;; Dimensions[EOSData][[1]], 1]]];
pressureEOS\[LetterSpace]m2[rhobar_] := 
 pressureEOS[
   rhobar/(MeVperfm3\[LetterSpace]2\[LetterSpace]Jperm3*
      Jperm3\[LetterSpace]2\[LetterSpace]m2)]*(MeVperfm3\[LetterSpace]\
2\[LetterSpace]Jperm3*Jperm3\[LetterSpace]2\[LetterSpace]m2)
dpressureEOS\[LetterSpace]m2[\[Rho]_] := 
  Evaluate[D[pressureEOS\[LetterSpace]m2[\[Rho]], \[Rho]]];
domainscaled = \
{mindensity*(MeVperfm3\[LetterSpace]2\[LetterSpace]Jperm3*
      Jperm3\[LetterSpace]2\[LetterSpace]m2), 
   maxdensity*(MeVperfm3\[LetterSpace]2\[LetterSpace]Jperm3*
      Jperm3\[LetterSpace]2\[LetterSpace]m2)};
MRfnc = Interpolation[MRDataLimited];





myscales = {10^5, 10^-10, 1};
MR\[LetterSpace]vals = 
  Table[RescaleMRPair\[LetterSpace]km\[LetterSpace]Msun[
    Quiet[GetGenericMassRadius[\[Rho]c, pressureEOS\[LetterSpace]m2, 
       10^-12, myscales][[1]]], myscales], {\[Rho]c, 3.2, 8.3, 0.1}];


Export["MR-vals-test.csv", MR\[LetterSpace]vals, "CSV"]
