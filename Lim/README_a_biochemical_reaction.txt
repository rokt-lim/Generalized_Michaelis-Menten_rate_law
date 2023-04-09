File name: a_biochemical_reaction.py


*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- "a_biochemical_reaction.py" requires Scipy 1.3.1 or greater,
Numpy 1.15.1 or greater, and pandas 1.0 or greater.


*** General information
This code is for the numerical simulation of the enzyme-substrate
reaction. The enzyme-substrate complex follows
dC(t)/dt = k_a [A(t) - C(t)]*[B(t) - C(t)] - k_delta C(t).
- C(t) is the concentration of the enzyme-substrate complex
at time t (unit = mM).
- A(t) and B(t) denote the total concentrations of
A (substrate, unit = mM) and B (enzyme, unit = mM).
- k_a denotes the association rate of free A and B (unit = mM^-1 sec^-1).
- k_delta is defined as k_delta = k_d + r_c.
- k_d is the dissociation rate of the enzyme-substrate complex (unit = sec^-1).
- r_c is the chemical conversion rate of A upon the formation of
the enzyme-substrate complex (unit = sec^-1).

The total concentration of A(t) changes over time as
dA(t)/dt = -r_c*C(t).

In the code, delta_tQ is defined as follows:
delta_tQ = (1 + (A + B)/K)^2 - 4*A*B/(K^2).


*** Main module
The main module of this code is "enzyme_substrate_reaction_simulation".
This module requires the following arguments:
- A_0     : the initial concentration of A (unit = mM),
- B_0     : the initial concentration of B (unit = mM),
- C_0     : the initial concentration of C (unit = mM),
- K_M     : the the Michaelis constant, K_M = k_delta/k_a (unit = mM),
- k_d     : the dissociation rate of the enzyme-substrate complex (unit = sec^-1),
- r_c     : the chemical conversion rate (unit = sec^-1),
- t_start : the start time of the simulation,
- t_end   : the end time of the simulation,
- dt      : the time step size of the simulation.
For the use of this module, "enzyme_substrate_reaction_simulation", please
refer "main_A_biochemical_reaction" module in the code.

The main module "enzyme_substrate_reaction_simulation" returns a dataframe
df_sol. This dataframe consists of the followings:
- the time points at which to store the computed solutions,
- the concentrations of A, B, C, C_sQSSA, C_tQSSA, and C_gamma at the time points,
- the values of epsilon_1, epsilon_2, epsilon_gamma, and epsilon_tQ
at the time points.
For the details on epsilon_1, epsilon_2, epsilon_gamma, and epsilon_tQ, please
see the supplementary material.


*** Demos
We provide the example module "main_A_biochemical_reaction" for running
"enzyme_substrate_reaction_simulation". The module, "main_A_biochemical_reaction"
is the numerical simulation for an enzyme-substrate reaction, when the substrate is
oxaloacetate and the enzyme is malate dehydrogenase. The simulation time interval
is (0.0 sec,0.005 sec) with a uniform time step size 0.000001 sec.
Running "main_A_biochemical_reaction.py" with the given
kinetic parameters and the settings takes less than 1 minute on a "normal desktop".


*** Contact information
This code was written by Roktaek Lim. If you have any question,
please contact R. Lim via "rokt.lim@gmail.com".
