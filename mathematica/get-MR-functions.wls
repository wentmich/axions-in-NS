(*Limits Mathematica to requested resources*)
Unprotect[$ProcessorCount];$ProcessorCount = 40;
SetOptions[$Output, FormatType->OutputForm];

(* PRECISION AND ACCURACY *)
NPRECISON = 13;

(* GLOBALS *)
Msun = 1.989*10^30;(* kg *)
GN = 6.67*10^-11; (* SI *)
G =(*6.70883*10^(-57)*)1; (* eV^-2 *)
clight = 2.99792458*10^8; (* m/s *)
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
RHOFRAC=10^(-8);
RSTART = 10^(-8);


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





(* ---------------------------------------------------------------------------------------------------------------------------------- *)
(* SOLVE FOR MASS FUNCTION, DENSITY FUNCTION, MASS, AND RADIUS GIVEN \
EQUATION OF STATE AND CENTRAL DENSITY *)
GetGenericMassRadius[\[Rho]central_, EOS_, dEOSdrhoofr_, \[Rho]frac_, 
  scales_] := 
 Module[{dMdr, dPdr, \[Rho]c = \[Rho]central, sols, Rstar, Mstar, 
   mass, density, rstart = RSTART, rend = 50.0, alpha, Acoef, 
   density\[LetterSpace]out, density\[LetterSpace]total, ddensitydr, 
   mass\[LetterSpace]out, mass\[LetterSpace]total, EOSscaled, 
   dEOSscaleddrhoofr, M, \[Rho], rt},
  (*EOSscaled[\[Rho]_]:=EOS[\[Rho]*scales[[2]]]/scales[[4]];*)
  (*dEOSdrhoofr[\[Rho]_]:= D[EOS[\[Rho]],\[Rho]];*)
  dMdr = 
   D[M[rt], rt] == 
    4*Pi*rt^2*\[Rho][rt]*(scales[[1]]^3* scales[[2]]/scales[[3]]);
  dPdr = 
   D[\[Rho][rt], 
     rt] == -((dEOSdrhoofr[scales[[2]]*\[Rho][rt]])^(-1))*(M[rt] + 
       4*Pi*rt^3*EOS[\[Rho][rt]*scales[[2]]]*
        scales[[1]]^3/scales[[3]])*(EOS[\[Rho][rt]*scales[[2]]]/
         scales[[2]] + \[Rho][
         rt])/(rt*(rt*scales[[1]]/scales[[3]] - 2*M[rt]));
  sols = 
   NDSolve[{{dMdr, 
       dPdr}, {M[rstart] == (4/3)*Pi*\[Rho]c*rstart^3*scales[[2]]*
         scales[[1]]^3/scales[[3]], \[Rho][
         rstart] == \[Rho]c - (2*Pi/3)*(\[Rho]c + 
            EOS[\[Rho]c*scales[[2]]]/scales[[2]])*(\[Rho]c*
             scales[[2]] + 3*EOS[\[Rho]c*scales[[2]]])*rstart^2*
          scales[[1]]^2, 
       WhenEvent[\[Rho][rt] == \[Rho]c*\[Rho]frac, {Rstar = rt, 
         Mstar = M[rt]}; "StopIntegration"]}}, {M, \[Rho]}, {rt, 
      rstart, rend}, PrecisionGoal -> NPRECISON, 
     AccuracyGoal -> NPRECISON][[1]];
  mass[rt_] := Evaluate[M[rt] /. sols];
  density[rt_] := Evaluate[\[Rho][rt] /. sols];
  ddensitydr[rt_] := Evaluate[D[density[rt], rt]];
  (*Rstar=Re[density["Domain"][[1]][[2]]];
  Mstar=Re[mass[Rstar]];*)
  alpha = -(ddensitydr[Rstar]/density[Rstar]);
  Acoef = density[Rstar];
  density\[LetterSpace]out[rt_] := Acoef*Exp[-alpha*(rt - Rstar)];
  density\[LetterSpace]total[rt_] := 
   If[rt <= Rstar, density[rt], density\[LetterSpace]out[rt]];
  mass\[LetterSpace]out[rt_] := 
   Mstar - ((Acoef*(-1 + Exp[alpha*(-rt + Rstar)]))/alpha);
  mass\[LetterSpace]total[rt_] := 
   If[rt <= Rstar, mass[rt], mass\[LetterSpace]out[rt]];
  {{Rstar, Mstar}, {mass\[LetterSpace]total, 
    density\[LetterSpace]total}}]



(* SOLVE FOR AXION AND AXION DERIVATIVE FIELDS *)
GetAxionField[massradiussol_, mass_, density_, ma_, fa_, scales_] := 
 Module[{density\[LetterSpace]coefficient\[LetterSpace]sub, 
   d\[Theta]pdr\[LetterSpace]rscaled, 
   d\[Theta]dr\[LetterSpace]rscaled, mythetaTfncsols, NSradiusscaled, 
   r, rt, \[Theta]sol, \[Theta]primesol, rstart = 10.0/10^5, 
   NSradius = 
    massradiussol[[
     1]], \[Theta]solin, \[Theta]solout, \[Theta]primesolin, \
\[Theta]primesolout},
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
      rstart, NSradius}, PrecisionGoal -> NPRECISON, 
     AccuracyGoal -> NPRECISON][[1]];
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
 Module[{vascaled, \[Rho]asol, pasol, rstart = 10.0/10^5, Masol, 
   NSradius = massradiussol[[1]], matotal, madiffeq, mainside, masols,
    rt, mainsol},
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
       rstart, NSradius}, PrecisionGoal -> NPRECISON, 
     AccuracyGoal -> NPRECISON][[1]];
  mainsol[rt_] := Evaluate[mainside[rt] /. masols];
  Masol[r_] := If[r < NSradius, mainsol[r], matotal];
  {{\[Rho]asol, pasol, Masol}}]



(* SOLVE FOR AXION METRIC PERTURBATIONS *)
GetDeltaAndZetaMetricPerturbations[massradiussol_, mass_, density_, 
  pressure_, axion\[LetterSpace]density_, 
  axion\[LetterSpace]pressure_, Matotal_, scales_] := 
 Module[{d\[CapitalDelta]dr, 
   d\[Zeta]dr, \[CapitalDelta]initial, \[Zeta]initial, 
   NSradius = massradiussol[[1]], NSmass = massradiussol[[2]], 
   rstart = 
    10.0/10^5, \[CapitalDelta]and\[Zeta]fncsols, \
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
     PrecisionGoal -> NPRECISON, AccuracyGoal -> NPRECISON][[1]];
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
GetTidalDeformabilityWithAxions[massradiussol_, M_, p_, \[Rho]_, 
  axionmassfnc_, pa_, \[Rho]a_, \[CapitalDelta]_, \[Zeta]_, scales_] :=
  Module[{yequqation, yinitial, yfncsols, yfncsol, 
   Rs = massradiussol[[1]], rstart = 10.0/10^5, 
   tidal\[LetterSpace]deformability, yatsfc, Co, sourceterm, yeq2, 
   Ma = axionmassfnc[massradiussol[[1]]], Mns = massradiussol[[2]], 
   tidal\[LetterSpace]deformability\[LetterSpace]noaxions, 
   mscale = scales[[3]], 
   rscale = scales[[1]], \[Rho]scale = scales[[2]], 
   tidal\[LetterSpace]deformability\[LetterSpace]axions, y0atsfc, 
   y1atsfc, y0, y1, y0eq, y1eq, sourceterm0, sourceterm1, y0initial, 
   y0fncsols, y0fncsol, y1initial, y1fncsols, y1fncsol},
  y0eq = y0'[r]/rscale == (-4*mscale^3*M[r]^3*(-1 + y0[r]^2) + 4*Pi*r^5*rscale^5*(p[r]*(6 + 64*Pi^2*r^4*rscale^4*p[r]^2 - y0[r] - y0[r]^2 - 4*Pi*r^2*rscale^2*p[r]*(9 + y0[r]) + 
      4*Pi*r^2*rscale^2*\[Rho]scale*(-5 + y0[r])*\[Rho][r]) + r*\[Rho]scale*Derivative[1][\[Rho]][r]) + 
  mscale*r^2*rscale^2*M[r]*(6 + 32*Pi^2*r^4*rscale^4*p[r]^2*(15 + y0[r]) - 20*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] - y0[r]*(1 + y0[r] - 4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) + 
    4*Pi*r^2*rscale^2*p[r]*(-21 + y0[r] + 4*y0[r]^2 - 8*Pi*r^2*rscale^2*\[Rho]scale*(-5 + y0[r])*\[Rho][r]) - 16*Pi*r^3*rscale^2*\[Rho]scale*Derivative[1][\[Rho]][r]) + 
  2*mscale^2*r*rscale*M[r]^2*(-6 + 4*Pi*r^2*rscale^2*p[r]*(15 + y0[r] - 2*y0[r]^2) + 20*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 
    y0[r]*(1 + 2*y0[r] - 4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) + 8*Pi*r^3*rscale^2*\[Rho]scale*Derivative[1][\[Rho]][r]))/
 (r*rscale*(r*rscale - 2*mscale*M[r])^2*(mscale*M[r] + 4*Pi*r^3*rscale^3*p[r]));
  sourceterm0[rt_] := Evaluate[y0eq /. {r -> rt}];
  y0initial = y0[rstart] == 2;
  y0fncsols = 
   NDSolve[{sourceterm0[rt], y0initial}, {y0}, {rt, rstart, Rs}, 
     PrecisionGoal -> NPRECISON, AccuracyGoal -> NPRECISON][[1]];
  y0fncsol[rt_] := Evaluate[y0[rt] /. y0fncsols];
  y0atsfc = y0fncsol[Rs];
  y1eq = y1'[r]/rscale ==(4*Pi*r^3*rscale^3*pa[r]*(mscale*r*rscale*M[r]*(9 - 24*Pi*r^2*rscale^2*p[r]*(-1 + y0fncsol[r]) - 5*y0fncsol[r]) + 2*mscale^2*M[r]^2*(-3 + y0fncsol[r]) + 
    2*r^2*rscale^2*(-1 + 32*Pi^2*r^4*rscale^4*p[r]^2 + y0fncsol[r] + 2*Pi*r^2*rscale^2*p[r]*(1 + 3*y0fncsol[r]))) - 
  2*mscale^3*M[r]^3*(-\[CapitalDelta][r] + y0fncsol[r]*(4*y1[r] + \[CapitalDelta][r] - \[Zeta][r]) + \[Zeta][r]) + 
  4*Pi*r^5*rscale^5*(32*Pi^2*r^4*rscale^4*p[r]^3*(\[CapitalDelta][r] + \[Zeta][r]) - 2*\[Rho]a[r] + 2*y0fncsol[r]*\[Rho]a[r] + 
    4*Pi*r^2*rscale^2*p[r]^2*(-y1[r] - 5*\[Zeta][r] - y0fncsol[r]*\[Zeta][r] + \[CapitalDelta][r]*(1 + 4*y0fncsol[r] + 16*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) + 16*Pi*r^2*rscale^2*\[Rho]a[r]) + 
    p[r]*(\[Zeta][r]*(2 - 12*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) + y1[r]*(-1 - 2*y0fncsol[r] + 4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) + 
      \[CapitalDelta][r]*(4 + y0fncsol[r] - 4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 4*Pi*r^2*rscale^2*\[Rho]scale*y0fncsol[r]*\[Rho][r]) + 4*Pi*r^2*rscale^2*(1 + y0fncsol[r])*\[Rho]a[r]) + 
    r*\[Rho]scale*\[CapitalDelta][r]*Derivative[1][\[Rho]][r] + r*\[Rho]scale*\[Zeta][r]*Derivative[1][\[Rho]][r] + r*Derivative[1][\[Rho]a][r]) + 
  mscale^2*r*rscale*M[r]^2*(-2*\[Zeta][r] + 48*Pi*r^2*rscale^2*p[r]*\[Zeta][r] - y0fncsol[r]*\[Zeta][r] + 16*Pi*r^2*rscale^2*p[r]*y0fncsol[r]*\[Zeta][r] + 
    24*Pi*r^2*rscale^2*\[Rho]scale*\[Zeta][r]*\[Rho][r] + y1[r]*(2 + 8*Pi*r^2*rscale^2*p[r]*(1 - 4*y0fncsol[r]) + 8*y0fncsol[r] - 8*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) - 
    24*Pi*r^2*rscale^2*\[Rho]a[r] + 24*Pi*r^2*rscale^2*y0fncsol[r]*\[Rho]a[r] + 16*Pi*r^3*rscale^2*\[Rho]scale*\[Zeta][r]*Derivative[1][\[Rho]][r] - 
    \[CapitalDelta][r]*(10 + y0fncsol[r] + 40*Pi*r^2*rscale^2*p[r]*y0fncsol[r] - 24*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 8*Pi*r^2*rscale^2*\[Rho]scale*y0fncsol[r]*\[Rho][r] - 
      16*Pi*r^3*rscale^2*\[Rho]scale*Derivative[1][\[Rho]][r]) + 16*Pi*r^3*rscale^2*Derivative[1][\[Rho]a][r]) + 
  mscale*r^2*rscale^2*M[r]*(y1[r]*(-1 + 32*Pi^2*r^4*rscale^4*p[r]^2 - 2*y0fncsol[r] + 4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 
      4*Pi*r^2*rscale^2*p[r]*(1 + 8*y0fncsol[r] - 8*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r])) + \[CapitalDelta][r]*(4 + y0fncsol[r] - 128*Pi^2*r^4*rscale^4*p[r]^2*y0fncsol[r] - 
      4*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 4*Pi*r^2*rscale^2*\[Rho]scale*y0fncsol[r]*\[Rho][r] - 4*Pi*r^2*rscale^2*p[r]*(9 - 40*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 
        y0fncsol[r]*(-3 + 8*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r])) - 16*Pi*r^3*rscale^2*\[Rho]scale*Derivative[1][\[Rho]][r]) + 
    2*\[Zeta][r]*(1 + 16*Pi^2*r^4*rscale^4*p[r]^2*(8 + y0fncsol[r]) - 6*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r] + 
      2*Pi*r^2*rscale^2*p[r]*(-7 - 2*y0fncsol[r] + 24*Pi*r^2*rscale^2*\[Rho]scale*\[Rho][r]) - 8*Pi*r^3*rscale^2*\[Rho]scale*Derivative[1][\[Rho]][r]) - 
    4*Pi*r^2*rscale^2*((-9 + 8*Pi*r^2*rscale^2*p[r]*(-3 + y0fncsol[r]) + 7*y0fncsol[r])*\[Rho]a[r] + 4*r*Derivative[1][\[Rho]a][r])))/
 (r*rscale*(r*rscale - 2*mscale*M[r])^2*(mscale*M[r] + 4*Pi*r^3*rscale^3*p[r])); 
  sourceterm1[rt_] := Evaluate[y1eq /. {r -> rt}];
  y1initial = 
   y1[rstart] == (2*(3*\[CapitalDelta][rstart] + \[Zeta][rstart])*(3*
           p[rstart] + \[Rho][rstart]*scales[[2]]) + 6*pa[rstart] + 
       6*\[Rho]a[
         rstart])/(5*(3*p[rstart] + \[Rho][rstart]*scales[[2]]));
  y1fncsols = 
   NDSolve[{sourceterm1[rt], y1initial}, {y1}, {rt, rstart, Rs}, 
     PrecisionGoal -> NPRECISON, AccuracyGoal -> NPRECISON][[1]];
  y1fncsol[rt_] := Evaluate[y1[rt] /. y1fncsols];
  y1atsfc = y1fncsol[Rs];
  (*Print[y0atsfc];*)
  tidal\[LetterSpace]deformability\[LetterSpace]noaxions = (8*Rs^2*
      rscale^2*(2*Mns*mscale - 
        Rs*rscale)*(4*Mns*
         mscale*(-3*Mns*mscale*Rs^6*rscale^6*(-2 + y0atsfc)^2 + 
           16*Mns^7*mscale^7*(-1 + y0atsfc^2) + 
           8*Mns^6*mscale^6*Rs*rscale*(5 - 4*y0atsfc + y0atsfc^2) + 
           3*Mns^2*mscale^2*Rs^5*
            rscale^5*(28 - 32*y0atsfc + 9*y0atsfc^2) - 
           4*Mns^5*mscale^5*Rs^2*
            rscale^2*(34 - 60*y0atsfc + 27*y0atsfc^2) + 
           2*Mns^4*mscale^4*Rs^3*
            rscale^3*(130 - 204*y0atsfc + 77*y0atsfc^2) - 
           Mns^3*mscale^3*Rs^4*
            rscale^4*(220 - 292*y0atsfc + 94*y0atsfc^2)) + 
        6*Mns*mscale*Rs^2*
         rscale^2*(2*Mns*mscale - 
            Rs*rscale)^3*(Rs*rscale*(-2 + y0atsfc) - 
            2*Mns*mscale*(-1 + y0atsfc))^2*
         Log[1 - (2*Mns*mscale)/(Rs*rscale)]))/(15*Mns*
      mscale*(2*Mns*
          mscale*(2*Mns^2*mscale^2*Rs^2*rscale^2*(13 - 11*y0atsfc) - 
            3*Rs^4*rscale^4*(-2 + y0atsfc) + 
            4*Mns^4*mscale^4*(1 + y0atsfc) + 
            2*Mns^3*mscale^3*Rs*rscale*(-2 + 3*y0atsfc) + 
            3*Mns*mscale*Rs^3*rscale^3*(-8 + 5*y0atsfc)) + 
         3*Rs^2*rscale^2*(-2*Mns*mscale + 
             Rs*rscale)^2*(-(Rs*rscale*(-2 + y0atsfc)) + 
            2*Mns*mscale*(-1 + y0atsfc))*
          Log[1 - (2*Mns*mscale)/(Rs*rscale)])^2);
  tidal\[LetterSpace]deformability\[LetterSpace]axions =(8*Rs^2*rscale^2*(2*Mns*mscale - Rs*rscale)*(4*Mns*mscale*(15*Ma*mscale*Rs^6*rscale^6*(-2 + y0atsfc)^2 - 3*Mns*mscale*Rs^5*rscale^5*(-2 + y0atsfc)*
      (Rs*rscale*(-2 + y0atsfc) + 2*Ma*mscale*(-36 + 23*y0atsfc)) + Mns^2*mscale^2*Rs^4*rscale^4*(3*Rs*rscale*(28 - 32*y0atsfc + 9*y0atsfc^2) + 
       2*Ma*mscale*(586 - 772*y0atsfc + 247*y0atsfc^2)) + 2*Mns^4*mscale^4*Rs^2*rscale^2*(Rs*rscale*(130 - 204*y0atsfc + 77*y0atsfc^2) + 
       Ma*mscale*(404 - 710*y0atsfc + 315*y0atsfc^2)) - Mns^3*mscale^3*Rs^3*rscale^3*(Rs*rscale*(220 - 292*y0atsfc + 94*y0atsfc^2) + 
       Ma*mscale*(1452 - 2252*y0atsfc + 841*y0atsfc^2)) + 8*Mns^6*mscale^6*(Ma*mscale*(9 + 2*y0atsfc - 11*y0atsfc^2) + Rs*rscale*(5 - 4*y0atsfc + y0atsfc^2 - 4*y1atsfc)) - 
     4*Mns^5*mscale^5*Rs*rscale*(Ma*mscale*(61 - 56*y0atsfc + 19*y0atsfc^2) + Rs*rscale*(34 - 60*y0atsfc + 27*y0atsfc^2 - 2*y1atsfc)) + 
     16*Mns^7*mscale^7*(-1 + y0atsfc^2 + 2*y1atsfc)) - 3*Ma*mscale*Rs^2*rscale^2*(2*Mns*mscale - Rs*rscale)^3*(Rs*rscale*(-2 + y0atsfc) - 2*Mns*mscale*(-1 + y0atsfc))^2*
    Log[1 - (2*Mns*mscale)/(Rs*rscale)]^2 - 2*Ma*mscale*(2*Mns*mscale - Rs*rscale)*(-3*Rs^6*rscale^6*(-2 + y0atsfc)^2 + 16*Mns^6*mscale^6*(-1 + y0atsfc^2) + 
     16*Mns^5*mscale^5*Rs*rscale*(2 - 2*y0atsfc + y0atsfc^2) + 6*Mns*mscale*Rs^5*rscale^5*(16 - 18*y0atsfc + 5*y0atsfc^2) - 
     6*Mns^2*mscale^2*Rs^4*rscale^4*(46 - 60*y0atsfc + 19*y0atsfc^2) + 8*Mns^3*mscale^3*Rs^3*rscale^3*(43 - 67*y0atsfc + 25*y0atsfc^2) - 
     4*Mns^4*mscale^4*Rs^2*rscale^2*(42 - 80*y0atsfc + 37*y0atsfc^2))*Log[Rs*rscale] + 2*Ma*mscale*(2*Mns*mscale - Rs*rscale)*
    (-3*Rs^6*rscale^6*(-2 + y0atsfc)^2 + 16*Mns^6*mscale^6*(-1 + y0atsfc^2) + 16*Mns^5*mscale^5*Rs*rscale*(2 - 2*y0atsfc + y0atsfc^2) + 
     6*Mns*mscale*Rs^5*rscale^5*(16 - 18*y0atsfc + 5*y0atsfc^2) - 6*Mns^2*mscale^2*Rs^4*rscale^4*(46 - 60*y0atsfc + 19*y0atsfc^2) + 
     8*Mns^3*mscale^3*Rs^3*rscale^3*(43 - 67*y0atsfc + 25*y0atsfc^2) - 4*Mns^4*mscale^4*Rs^2*rscale^2*(42 - 80*y0atsfc + 37*y0atsfc^2))*Log[-2*Mns*mscale + Rs*rscale] + 
   6*Rs^2*rscale^2*(2*Mns*mscale - Rs*rscale)^3*(Rs*rscale*(-2 + y0atsfc) - 2*Mns*mscale*(-1 + y0atsfc))^2*Log[1 - (2*Mns*mscale)/(Rs*rscale)]*
    (-4*Ma*mscale + Mns*mscale - Ma*mscale*Log[Rs*rscale] + Ma*mscale*Log[-2*Mns*mscale + Rs*rscale])))/
 (15*Mns*mscale*(2*Mns*mscale*(2*Mns^2*mscale^2*Rs^2*rscale^2*(13 - 11*y0atsfc) - 3*Rs^4*rscale^4*(-2 + y0atsfc) + 4*Mns^4*mscale^4*(1 + y0atsfc) + 
      2*Mns^3*mscale^3*Rs*rscale*(-2 + 3*y0atsfc) + 3*Mns*mscale*Rs^3*rscale^3*(-8 + 5*y0atsfc)) + 3*Rs^2*rscale^2*(-2*Mns*mscale + Rs*rscale)^2*
     (-(Rs*rscale*(-2 + y0atsfc)) + 2*Mns*mscale*(-1 + y0atsfc))*Log[1 - (2*Mns*mscale)/(Rs*rscale)])^2); 
  {{tidal\[LetterSpace]deformability\[LetterSpace]axions, 
    tidal\[LetterSpace]deformability\[LetterSpace]noaxions}, \
{y0fncsol, y1fncsol}}]




(* ---------------------------------------------------------------------------------------------------------------------------------- *)
Print["Defined all functions"]
Pause[5]


MRPATH = "/home/wentzel4/axions-in-NS/EOS-MR/input_stable_eos_files_p_of_nb_fixed/"; 
files = FileNames["*-1.csv", MRPATH];

Print[files[[1]]];


myscales = {10^5, 10^-10, 1};
mafasubs = {10^ToExpression[$CommandLine[[-2]]], 10^ToExpression[$CommandLine[[-1]]]};
Npoints = ToExpression[$CommandLine[[-3]]];
downsamplefactor = ToExpression[$CommandLine[[-4]]];
shift = 3.1;
Module[{j, EOSData, EOSDataLimited, centraldensities, pressureEOS, 
  dpressureEOS, pressureEOS\[LetterSpace]m2, 
  dpressureEOS\[LetterSpace]m2}, EOSData = Import[files[[1]], "CSV"];
 EOSDataLimited = EOSData[[1 ;; -1 ;; downsamplefactor]];
 Print[EOSDataLimited];
 pressureEOS = 
  ResourceFunction["CubicSplineInterpolation"][EOSDataLimited];
 dpressureEOS[\[Rho]_] := Evaluate[D[pressureEOS[\[Rho]], \[Rho]]];
 pressureEOS\[LetterSpace]m2[rhobar_] := 
  pressureEOS[
    rhobar/(MeVperfm3\[LetterSpace]2\[LetterSpace]Jperm3*
       Jperm3\[LetterSpace]2\[LetterSpace]m2)]*(MeVperfm3\
\[LetterSpace]2\[LetterSpace]Jperm3*
     Jperm3\[LetterSpace]2\[LetterSpace]m2);
 dpressureEOS\[LetterSpace]m2[\[Rho]_] := 
  Evaluate[D[pressureEOS\[LetterSpace]m2[\[Rho]], \[Rho]]];
 Print["did everything pre do loop"];
 Pause[5];
 output\[LetterSpace]table\[LetterSpace]lots = 
  ParallelTable[
   Module[{\[Rho]c, mrm\[Rho]\[LetterSpace]vals, MNS, RNS, rhocNS, 
     massfnc, densityfnc, pressurefnc, 
     thetathetaprime\[LetterSpace]vals, thetafnc, thetafncprime, 
     axiondensitypressuremass\[LetterSpace]fncs, axiondensityfnc, 
     axionpressurefnc, axionmassfnc, Maxion, 
     deltaandzeta\[LetterSpace]fncs, zetafnc, deltafnc, 
     tidal\[LetterSpace]deformabilities, LambdaNS, 
     LambdaAxion}, \[Rho]c = 
     10^(Log10[3.3 - shift] + 
         j*(Log10[10 - shift] - Log10[3.3 - shift])/Npoints) + shift;
    mrm\[Rho]\[LetterSpace]vals = 
     Quiet[GetGenericMassRadius[\[Rho]c, pressureEOS\[LetterSpace]m2, 
       dpressureEOS\[LetterSpace]m2, RHOFRAC, myscales]];
    Print[mrm\[Rho]\[LetterSpace]vals[[1]]];
    MNS = mrm\[Rho]\[LetterSpace]vals[[1]][[2]]*myscales[[3]];
    RNS = mrm\[Rho]\[LetterSpace]vals[[1]][[1]]*myscales[[1]];
    rhocNS = \[Rho]c*myscales[[2]];
    massfnc[r_] := 
     Evaluate[mrm\[Rho]\[LetterSpace]vals[[2]][[1]][r]];
    densityfnc[r_] := 
     Evaluate[mrm\[Rho]\[LetterSpace]vals[[2]][[2]][r]];
    pressurefnc[r_] := 
     pressureEOS\[LetterSpace]m2[densityfnc[r]*myscales[[2]]];
    densityvals = 
  Quiet[Table[{r*myscales[[1]], densityfnc[r]*myscales[[2]]}, {r, 
     RSTART, 10*mrm\[Rho]\[LetterSpace]vals[[1]][[1]], 
     mrm\[Rho]\[LetterSpace]vals[[1]][[1]]/1000}]];
    Print["Got density values"];
    pressurevals = 
  Quiet[Table[{r*myscales[[1]], pressurefnc[r]}, {r, RSTART, 
     10*mrm\[Rho]\[LetterSpace]vals[[1]][[1]], 
     mrm\[Rho]\[LetterSpace]vals[[1]][[1]]/1000}]];
    massvals = 
  Quiet[Table[{r*myscales[[1]], massfnc[r]*myscales[[3]]}, {r, 
     RSTART, 10*mrm\[Rho]\[LetterSpace]vals[[1]][[1]], 
     mrm\[Rho]\[LetterSpace]vals[[1]][[1]]/1000}]];
    nuvals = 
  Quiet[Table[{r*myscales[[1]], nufnc[r]}, {r, RSTART, 
     10*mrm\[Rho]\[LetterSpace]vals[[1]][[1]], 
     mrm\[Rho]\[LetterSpace]vals[[1]][[1]]/1000}]];
    Export[StringJoin["/home/wentzel4/axions-in-NS/axions-in-NS-debora-EOS-1/density-vals-rhoc-", ToString[\[Rho]c], "-.csv"], densityvals, "CSV"];
    Export[StringJoin["/home/wentzel4/axions-in-NS/axions-in-NS-debora-EOS-1/pressure-vals-rhoc-", ToString[\[Rho]c], "-.csv"], pressurevals, "CSV"];
    Export[StringJoin["/home/wentzel4/axions-in-NS/axions-in-NS-debora-EOS-1/mass-vals-rhoc-", ToString[\[Rho]c], "-.csv"], massvals, "CSV"];
    Export[StringJoin["/home/wentzel4/axions-in-NS/axions-in-NS-debora-EOS-1/nu-vals-rhoc-", ToString[\[Rho]c], "-.csv"], nuvals, "CSV"];
    Export[StringJoin["/home/wentzel4/axions-in-NS/axions-in-NS-debora-EOS-1/radius-mass-vals-rhoc-", ToString[\[Rho]c], "-.csv"], {mrm\[Rho]\[LetterSpace]vals[[1]][[1]]*myscales[[1]], mrm\[Rho]\[LetterSpace]vals[[1]][[2]]*myscales[[3]]}, "CSV"];
     StringJoin["Working on step ", ToString[j], " of ", 
      ToString[Npoints]]; {rhocNS, RNS, MNS}], {j, 0, Npoints}]]

Export["/home/wentzel4/axions-in-NS/data/debora-EOS-1-runs/output-MR-curve.csv", output\[LetterSpace]table\[LetterSpace]lots, "CSV"]
(* Print[output\[LetterSpace]table\[LetterSpace]lots] *)
(* Export[StringJoin["/home/wentzel4/axions-in-NS/data/debora-EOS-1-runs/output-",ToString[ToString[mafasubs[[1]], FortranForm]],"-",ToString[ToString[mafasubs[[2]], FortranForm]],".csv" ], output\[LetterSpace]table\[LetterSpace]lots, "CSV"] *)

Print["Completed Job!!!"]
