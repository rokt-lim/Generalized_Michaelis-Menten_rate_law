File name: b_ppi_gi_ztl.py


*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- "b_ppi_gi_ztl.py" requires Scipy 1.3.1 or greater,
Numpy 1.15.1 or greater, and pandas 1.0 or greater.
- "b_ppi_gi_ztl.py" requires "ddeint.py" from "https://github.com/Zulko/ddeint".
- "b_ppi_gi_ztl.py" requires data files, "normalized_GI_data.csv" and
"normalized_ZTL_data.csv". "normalized_GI_data.csv" and "normalized_ZTL_data.csv"
are the experimental profiles of GI and ZTL, respectively.


*** General information
This code is for the numerical simulation of
the ZEITLUPE (ZTL)-GIGANTEA (GI) interaction. This interaction follows
dC(t)/dt = k_a [A(t) - C(t)]*[B(t) - C(t)] - k_delta C(t).
- C(t) is the concentration of GI-binding ZTL at time t (unit = nM).
- A(t) and B(t) denote the total concentrations of ZTL (unit = nM) and
GI (unit = nM).
- k_a denotes the association rate of free ZTL and free GI (unit = nM^-1 hour^-1).
- k_delta is defined as k_delta = k_d + r_c.
- k_d is the dissociation rate of GI-binding ZTL (unit = hour^-1).
- r_c is the degradation rate of GI-binding ZTL (unit = hour^-1).
- k_a can be computed by using k_a = k_delta/K where K (unit = nM) is
the Michaelis constant.
- Because blue light enhances the ZTL–GI interaction, we assume that k_d and K in light
do not exceed k_d and K in darkness, respectively.
- k_d_D >= k_d_L; k_d_L is the dissociation rate of GI-binding ZTL in light and
k_d_D is the dissociation rate of GI-binding ZTL in darkness.
- K_D >= K_L; K_L is the Michaelis constant in light and
K_D is the Michaelis constant in darkness.

The ZTL (A(t)) turnover dynamics can be described by the following equation:
dA(t)/dt = g_A(t) - r_c*C(t) - r_A*[A(t) - C(t)].
- g_A is the ZTL synthesis rate (unit = nM hour^-1).
- r_A is the degradation rate of free ZTL (unit = hour^-1).

In the code, delta_tQ is defined as follows:
delta_tQ = (K + A + B)^2 - 4*A*B.


*** Main module
The main modules of this code is "GI_ZTL_full", "GI_ZTL_gamma", and "GI_ZTL_tQSSA".

"GI_ZTL_full" module requires the following arguments:
- ZTL_data_file : file name of normalized protein level of ZTL,
- GI_data_file  : file name of normalized protein level of GI,
- w_ZTL         : scaling coefficient of ZTL (unit = nM),
- w_GI          : scaling coefficient of GI (unit = nM),
- g_A           : ZTL synthesis rate (unit = nM hour^-1),
- r_A           : degradation rate of free ZTL (unit = hour^-1),
- r_c           : degradation rate of GI-binding ZTL (unit = hour^-1)
- k_d_L         : dissociation rate of GI-binding ZTL in light (unit = hour^-1),
- k_d_D         : dissociation rate of GI-binding ZTL in darkness (unit = hour^-1),
- K_L           : the Michaelis constant in light (unit =nM),
- K_D           : the Michaelis constant in darkness (unit = nM),
- t_start       : the start time of the simulation,
- t_end         : the end time of the simulation,
- dt            : the time step size of the simulation.
"GI_ZTL_full" computes
- the time points at which to store the computed solutions,
- the normalized protein levels of GI at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t evaluated by solving the system of ODEs
dC(t)/dt = k_a [A(t) - C(t)]*[B(t) - C(t)] - k_delta C(t) and
dA(t)/dt = g_A(t) - r_c*C(t) - r_A*[A(t) - C(t)].

"GI_ZTL_gamma" module requires the following arguments:
- ZTL_data_file : file name of normalized protein level of ZTL,
- GI_data_file  : file name of normalized protein level of GI,
- w_ZTL         : scaling coefficient of ZTL (unit = nM),
- w_GI          : scaling coefficient of GI (unit = nM),
- g_A           : ZTL synthesis rate (unit = nM hour^-1),
- r_A           : degradation rate of free ZTL (unit = hour^-1),
- r_c           : degradation rate of GI-binding ZTL (unit = hour^-1)
- k_d_L         : dissociation rate of GI-binding ZTL in light (unit = hour^-1),
- k_d_D         : dissociation rate of GI-binding ZTL in darkness (unit = hour^-1),
- K_L           : the Michaelis constant in light (unit =nM),
- K_D           : the Michaelis constant in darkness (unit = nM),
- t_start       : the start time of the simulation,
- t_end         : the end time of the simulation,
- dt            : the time step size of the simulation.
"GI_ZTL_gamma" computes
- the time points at which to store the computed solutions,
- the normalized protein levels of GI at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t evaluated by solving
dA(t)/dt = g_A(t) - r_c*C_gamma(t) - r_A*[A(t) - C_gamma(t)].
Here, C_gamma(t) is the approximant of C(t). C_gamma(t) is proposed in this study.

"GI_ZTL_tQSSA" module requires the following arguments:
- ZTL_data_file : file name of normalized protein level of ZTL,
- GI_data_file  : file name of normalized protein level of GI,
- w_ZTL         : scaling coefficient of ZTL (unit = nM),
- w_GI          : scaling coefficient of GI (unit = nM),
- g_A           : ZTL synthesis rate (unit = nM hour^-1),
- r_A           : degradation rate of free ZTL (unit = hour^-1),
- r_c           : degradation rate of GI-binding ZTL (unit = hour^-1)
- K_L           : the Michaelis constant in light (unit =nM),
- K_D           : the Michaelis constant in darkness (unit = nM),
- t_start       : the start time of the simulation,
- t_end         : the end time of the simulation,
- dt            : the time step size of the simulation.
"GI_ZTL_tQSSA" computes
- the time points at which to store the computed solutions,
- the normalized protein levels of GI at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t using spline representation (degree = 3),
- the normalized protein levels of ZTL at time t evaluated by solving
dA(t)/dt = g_A(t) - r_c*C_tQSSA(t) - r_A*[A(t) - C_tQSSA(t)].
Here, C_tQSSA(t) is the approximant of C(t). C_tQSSA(t) is obtained by using
the total quasi-steady state approximation.

For the use of these modules, "GI_ZTL_full", "GI_ZTL_gamma", and "GI_ZTL_tQSSA", please
refer "main_B_protein_protein_interaction_GI_ZTL" module in the code.


*** Demos
We provide the example module "main_B_protein_protein_interaction_GI_ZTL" for
running "GI_ZTL_full", "GI_ZTL_gamma", and "GI_ZTL_tQSSA". The module,
"main_B_protein_protein_interaction_GI_ZTL" is the numerical simulation for
the ZTL-GI interaction. The kinetic parameters for the simulation are given
in "main_B_protein_protein_interaction_GI_ZTL". The simulation time interval is
(0 hour,480 hour) with a uniform time step size 0.05 hour.
Running "main_B_protein_protein_interaction_GI_ZTL.py" with the given
kinetic parameters and the settings takes less than 1 minute on a "normal desktop".


*** Contact information
This code was written by Roktaek Lim. If you have any question,
please contact R. Lim via "rokt.lim@gmail.com".
