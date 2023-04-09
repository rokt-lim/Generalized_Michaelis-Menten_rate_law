File name: iv_parameter_estimation.py


*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- "c_tf_dna_interaction.py" requires Scipy 1.5.1 or greater,
Numpy 1.15.1 or greater, and pandas 1.0 or greater.


*** General information
This code is for parameter estimations from the revised Michaelis-Menten law
and the total quasi-steady state approximation.
We consider protein–protein interactions with time-varying protein concentrations
A(t) and B(t); and TF–DNA interactions with time-varying TF concentration A_TF(t).
For the protein–protein interactions, we generate the data set as follows:
- Solve dC(t)/dt = k_a [A(t) - C(t)]*[B(t) - C(t)] - k_delta C(t) with
A(t) = A_max*{1 - alpha_A/2*[1 + cos(2*pi/T*t)]} and
B(t) = B_max*{1 - alpha_B/2*[1 + cos(2*pi/T*t - phi_B)]} in the time interval
(0.0 hour, 480.0 hour),
- Collect values of C(t) at t=[408,410,412,…,478,480] (total 37 points).
Then, we estimate K value from the tQSSA and estimate K and k_delta from
the revised Michaelis-Menten law. Powell's method is used for the parameter
estimation. In the case of the tQSSA, we estimate K by using
the Trust Region Reflective algorithm (gradient-based method).
We generate ten random initial guesses for the parameter estimations
(K for the tQSSA, K and k_delta for the revised Michaelis-Menten law).
Then, we choose the estimated parameter whose corresponding objective function
value is smallest.
In the code, delta_tQ is defined as follows:
delta_tQ = (K + A + B)^2 - 4*A*B.

For the TF–DNA interactions, we generate the data set as follows:
- Solve dC_TF(t)/dt = k_a*A_TF(t)*[1/V - C_TF(t)] - k_delta*C_TF(t) with
A_TF(t) = A_TF_max*{1 - alpha_A/2*[1 + cos(2*pi/T*t)]} in the time interval
(0.0 hour, 480.0 hour),
- Collect values of C_TF(t) at t=[408,410,412,…,478,480] (total 37 points).
Then, we estimate K value from the QSSA and estimate K and k_delta from
the revised Michaelis-Menten law. Powell's method is used for the parameter
estimation. In the case of the QSSA, we estimate K by using
the Trust Region Reflective algorithm (gradient-based method).
We generate ten random initial guesses for the parameter estimations
(K for the QSSA, K and k_delta for the revised Michaelis-Menten law).
The ranges for K and k_delta are 0.01 <= K <= 1000.0 and 0.1 <= k_delta <= 10.
Then, we choose the estimated parameter whose corresponding objective function
value is smallest.


*** Main module
The main modules of this code are "parameter_estimation_ppi" and
"parameter_estimation_tf".

"parameter_estimation_ppi" module requires the following arguments:
- A_bar_max   : the peak level of A_bar(tau) (dimensionless),
- alpha_A_bar : the peak-to-trough difference of A_bar(tau) (dimensionless),
- B_bar_max   : the peak level of B_bar(tau) (dimensionless),
- alpha_B_bar : the peak-to-trough difference of B_bar(tau) (dimensionless),
- phi_B       : the phase difference between A_bar and B_bar (dimensionless),
- K_M         : the Michaelis constant (unit = nM),
- k_delta     : k_delta = k_d + r_c + k_loc + k_dlt (unit = hour-1),
- t_data      : the time points for the values of C(t),
- C_data      : the values of C(t) at t in t_data,
- T_period    : the the oscillation period of A(t) and B(t) (in this simulation T=24 hour),
- t_start     : the start time for the data generation,
- t_end       : the end time for the data generation,
- dt          : the time step size for the data generation,
- df_p_init   : a dataframe containing ten random initial guesses.
k_d, k_loc, and k_dlt stand for the dissociation, translocation, and dilution rates of
the complex AB, respectively, and r_c for the chemical conversion or translocation rate
of A or B upon the formation of the complex AB.

"parameter_estimation_ppi" computes
- the best estimate of K from the tQSSA using Powell's method,
- the best estimate of K from the tQSSA using the Trust Region Reflective algorithm,
- the best estimates of K and k_delta from the revised Michaelis-Menten law
using Powell's method.


"parameter_estimation_tf" module requires the following arguments:
- A_max     : the peak level of A_TF(t) (unit = nM),
- alpha_A   : the peak-to-trough difference of A_TF(t) (dimensionless),
- V         : nuclear volume (unit = nM^-1),
- K_M       : the Michaelis constant (unit = nM),
- k_delta   : the TF–DNA unbinding rate (unit = hour^-1),
- t_data    : the time points for the values of C_TF(t),
- C_data    : the values of C_TF(t) at t in t_data,
- T_period  : the the oscillation period of A_TF(t) (in this simulation T=24 hour),
- t_start   : the start time for the data generation,
- t_end     : the end time for the data generation,
- dt        : the time step size for the data generation,
- df_p_init : a dataframe containing ten random initial guesses.
"solve_TF_DNA_interaction_irregular" computes
- the time points at which to store the computed solutions,

"parameter_estimation_tf" computes
- the best estimate of K from the QSSA using Powell's method,
- the best estimate of K from the QSSA using the Trust Region Reflective algorithm,
- the best estimates of K and k_delta from the revised Michaelis-Menten law
using Powell's method.


*** Demos
We provide the example module "main_IV_p_estimation_ppi" and "main_IV_p_estimation_tf"
for running "parameter_estimation_ppi" and "parameter_estimation_tf", respectively.
The modules, "main_IV_p_estimation_ppi" and "main_IV_p_estimation_tf" are
the parameter estimations for the protein–protein interactions and the TF–DNA interactions,
respectively. The kinetic parameters for the parameter estimations are
given in "main_IV_p_estimation_ppi" and "main_IV_p_estimation_tf".
The time intervals for the data generation are (0.0 hour,480 hour)
with a uniform time step size 0.05 hour.
Running "main_IV_p_estimation_ppi" and "main_IV_p_estimation_tf" with the given
kinetic parameters and the settings take about 20 minutes on a "normal desktop".


*** Contact information
This code was written by Roktaek Lim. If you have any question,
please contact R. Lim via "rokt.lim@gmail.com".
