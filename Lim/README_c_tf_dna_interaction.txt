File name: c_tf_dna_interaction.py


*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- "c_tf_dna_interaction.py" requires Scipy 1.3.1 or greater,
Numpy 1.15.1 or greater, and pandas 1.0 or greater.


*** General information
This code is for the numerical simulation of
the Transcription factor (TF)-DNA interaction. This interaction follows
dC_TF_bar(tau)/d(tau) = A_TF_bar(tau)/(K*V) - [1 + A_TF_bar(tau)]*C_TF_bar(tau).
- tau = k_delta*t.
- k_delta is the TF–DNA unbinding rate (unit = hour^-1).
- C_TF_bar is the dimensionless quantity proportional to
the concentration of DNA-binding TF (C_TF(t), unit = nM).
- A_TF_bar is the dimensionless quantity proportional to
the total TF concentration (A_TF(t), unit = nM).
- K (unit = nM) is the Michaelis constant (unit = nM).
- k_a (TF–DNA binding rate, unit = nM^-1 hour^-1) is computed as k_a = k_delta/K
- V is the nuclear volume (nM^-1).

The TF concentration oscillates over time in a sinusoidal form:
A_TF_bar(tau) = A_TF_bar_max*{1 - alpha_A/2*[1 + cos(2*pi/(k_delta*T)*tau)]}
- A_TF_bar_max is the peak level of A_TF_bar.
- alpha_A is the peak-to-trough difference of A_TF_bar.
- T is the oscillation period of A_TF (in this simulation T=24 hour).


*** Main module
The main modules of this code are "solve_TF_DNA_interaction" and
"solve_TF_DNA_interaction_irregular".

"solve_TF_DNA_interaction" module requires the following arguments:
- A_max    : the peak level of a transcription factor protein, A_TF(t) (unit = nM),
- alpha_A  : the peak-to-trough difference of A_TF(t) (dimensionless),
- K_M      : the Michaelis constant (unit = nM),
- V        : nuclear volume (unit = nM^-1)
- k_delta  : the TF–DNA unbinding rate (unit = hour^-1),
- T_period : the the oscillation period of A_TF (in this simulation T=24 hour),
- t_start  : the start time of the simulation,
- t_end    : the end time of the simulation,
- dt       : the time step size of the simulation.
"solve_TF_DNA_interaction" computes
- the time points at which to store the computed solutions,
- the concentrations of A_TF, C_TF, C_TF_QSSA, and C_TF_gamma at the time points,
- the values of epsilon_TF, epsilon_TF_gamma, and epsilon_TF_Q at the time points.
For the details on epsilon_TF, epsilon_TF_gamma, and epsilon_TF_Q, please
see the supplementary material.

"solve_TF_DNA_interaction_irregular" computes the solution of
dC_TF_bar(tau)/d(tau) = A_TF_bar(tau)/(K*V) - [1 + A_TF_bar(tau)]*C_TF_bar(tau)
using
A_TF(t) = 1/N*(sum_{i=1}^{N} A_TF_max,i*{1 - alpha_A,i/2*[1 + cos(2*pi/T_A,i*t - phi_A,i)]}), N=10.
This module requires the following arguments:
- A_max   : the values of A_TF_max,i (unit = nM),
- alpha_A : the values of alpha_A,i (dimensionless),
- K_M     : the Michaelis constant (unit = nM),
- V       : nuclear volume (unit = nM^-1),
- k_delta : the TF–DNA unbinding rate (unit = hour^-1),
- phi_A   : the values of phi_A,i (dimensionless),
- T_A     : the values of T_A,i (unit = hour),
- t_start : the start time of the simulation,
- t_end   : the end time of the simulation,
- dt      : the time step size of the simulation.
"solve_TF_DNA_interaction_irregular" computes
- the time points at which to store the computed solutions,
- the concentrations of A_TF, C_TF, C_TF_QSSA, and C_TF_gamma at the time points.

For the use of these modules, "solve_TF_DNA_interaction" and
"solve_TF_DNA_interaction_irregular", please refer 
"main_C_TF_DNA_interaction" and "main_C_TF_DNA_interaction_irregular" modules in the code.


*** Demos
We provide the example module "main_C_TF_DNA_interaction" and "main_C_TF_DNA_interaction_irregular"
for running "solve_TF_DNA_interaction" and "solve_TF_DNA_interaction_irregular", respectively.
The modules, "main_C_TF_DNA_interaction" and "main_C_TF_DNA_interaction_irregular" are
the numerical simulations for the TF-DNA interaction. The kinetic parameters for the simulations are
given in "main_C_TF_DNA_interaction" and "main_C_TF_DNA_interaction_irregular".
The simulation time intervals are (0 hour,480 hour) with a uniform time step size 0.05 hour.
Running "main_C_TF_DNA_interaction" and "main_C_TF_DNA_interaction_irregular" with the given
kinetic parameters and the settings take less than 1 minute on a "normal desktop".


*** Contact information
This code was written by Roktaek Lim. If you have any question,
please contact R. Lim via "rokt.lim@gmail.com".
