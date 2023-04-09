File name: ComputeFunctionsAndEpsilon.py

*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- "ComputeFunctionsAndEpsilon.py" requires Scipy 1.1.0 or greater,
Numpy 1.15.1 or greater, and Matplotlib 2.2.3 or greater.

*** General information
This file contains functions to compute the real solution of the equation:
dC_bar(tau)/dtau = [A_bar(tau) - C_bar(tau)][B_bar(tau) - C_bar(tau)] - C_bar(tau)

*** Main modules
The forms of A_bar and B_bar are given by:
A_bar(tau) = A_bar_max{1 - alpha_A/2[1 + cos(2pi.tau/k_deltaT)]}
B_bar(tau) = B_bar_max{1 - alpha_B/2[1 + cos(2pi.tau/k_deltaT - phi_B)]}

The function RandomGenerate takes 7 parameters as input:
-N : Integer, number of random sets of parameters
-t0, tf : real numbers, boundaries of the output arrays
-t0eval, tfeval : real numbers, interval to solve the ODE
-dt : real number, time step for the outputs arrays 
-T : real number, time period for the functions A_bar and B_bar

This function computes and stores in .npy files the values of A_bar, B_bar, C_full, C_gamma, C_tQ, C_sQ, Eps_1, Eps_2, Eps_gamma and Eps_tQ from t0 to tf with dt time step.
Each step of parameters (A_max, B_max, k_delta, alpha_A, alpha_B, phi_B) are also store in .npy file.

The function Draw_Individual takes 12 parameters as input: 
-(A_max, B_max, k_delta, alpha_A, alpha_B, phi_B) : real numbers, the parameters for A_bar and B_bar
-t0, tf : real numbers, boundaries of the output arrays
-t0eval, tfeval : real numbers, interval to solve the ODE
-dt : real number, time step for the outputs arrays 
-T : real number, time period for the functions A_bar and B_bar

Outputs are three graphs of time series between t0 and tf, first one with A_bar, B_bar and C_full, second one with C_full, C_tQ and C_sQ and third one with C_full and C_gamma.

The second part of this file is about the irregular case for A_bar and B_bar with the forms:
A_bar(tau) = 1/N Sum{i=1 to N}(A_bar_max(i){1 - alpha_A(i)/2[1 + cos(2pi.tau/k_deltaT_A(i) - phi_A(i))]}
B_bar(tau) = 1/N Sum{i=1 to N}(B_bar_max(i){1 - alpha_B(i)/2[1 + cos(2pi.tau/k_deltaT_B(i) - phi_B(i))]}

The function RandomGenerate_Irregular takes 6 parameters as input : 
-N : Integer, number of random sets of parameters
-t0, tf : real numbers, boundaries of the output arrays
-t0eval, tfeval : real numbers, interval to solve the ODE
-dt : real number, time step for the outputs arrays 

For each set of parameters, it prints the full set of parameters (List of 10 A_bar_max, List of 10 B_bar_max, k_delta, List of 10 alpha_A, List of 10 alpha_B, List of 10 phi_A, List of 10 phi_B, List of 10 TA, List of 10 TB) and draws 3 graphs of time series between t0 and tf, first one with A_bar, B_bar and C_full, second one with C_full, C_tQ and C_sQ and third one with C_full and C_gamma.

The function Draw_Irregular takes 14 paremeters as input :
-(L_A_max, L_B_max) : list of N values for A_bar and B_bar
-k_delta : real number for A_bar and B_bar
-(L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) : list of N numbers for A_bar and B_bar
-t0, tf : real numbers, boundaries of the output arrays
-t0eval, tfeval : real numbers, interval to solve the ODE 
-dt : real number, time step for the outputs arrays 

Outputs are three graphs of time series between t0 and tf, first one with A_bar, B_bar and C_full, second one with C_full, C_tQ and C_sQ and third one with C_full and C_gamma.

*** Contact information
This code was written by Thomas L. P. Martin. If you have any question,
please contact T. Martin via "thomas.martin.stx@gmail.com".