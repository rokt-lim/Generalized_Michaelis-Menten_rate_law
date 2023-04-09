File name: e_rhythmic_degradation_n1.py
           e_rhythmic_degradation_n2.py
           e_rhythmic_degradation_n3.py


*** System requirement
- Windows, Linux or macOS
- Python 3.7 or greater
- The python codes require Scipy 1.3.1 or greater,
Numpy 1.15.1 or greater, and pandas 1.0 or greater.


*** General information
These codes are for the numerical simulations of the rhythmic degradation of proteins.
The degradation mechanism requires post-translational modifications (PTMs).
In the simulations, A_0(t) denotes the concentration of unmodified protein and
A_i(t) denotes the concentration of the i-th modified protein with i=1,2,...,n
(n is the total number of PTMs). Let A(t) be the total concentration of protein,
A(t) = sum_{i=0}^{n} A_i(t).

The protein turnover dynamics for n=1,2,3 are given as follows:

n = 1
dA_0(t)/dt = g(t) - a_0*A_0(t)
dA_1(t)/dt = a_0*A_0(t) - r_c*A_1(t)

n = 2
dA_0(t)/dt = g(t) - a_0*A_0(t)
dA_1(t)/dt = a_0*A_0(t) - a_1*A_1(t)
dA_2(t)/dt = a_1*A_1(t) - r_c*A_2(t)

n = 3
dA_0(t)/dt = g(t) - a_0*A_0(t)
dA_1(t)/dt = a_0*A_0(t) - a_1*A_1(t)
dA_2(t)/dt = a_1*A_1(t) - a_2*A_2(t)
dA_3(t)/dt = a_2*A_2(t) - r_c*A_3(t)

where g(t) is the protein synthesis (translation) rate and
a_i, i=0,...,n-1 denotes the rate of the (i+1)-th modification
with phosphorylation or ubiquitination, and
r_c is the modified protein's turnover rate.


The protein turnover dynamics for n=1,2,3 can be rewritten as follows:

n = 1
dA(t)/dt = g(t) - r_c*A_1(t)
dA_1(t)/dt = a_0*(A(t) - A_1(t)) - r_c*A_1(t)

n = 2
dA(t)/dt   = g(t) - r_c*A_2(t)
dA_1(t)/dt = a_0*(A(t) - A_1(t) - A_2(t)) - a_1*A_1(t)
dA_2(t)/dt = a_1*A_1(t) - r_c*A_2(t)

n = 3
dA(t)/dt   = g(t) - r_c*A_3(t)
dA_1(t)/dt = a_0*(A(t) - A_1(t) - A_2(t) - A_3(t)) - a_1*A_1(t)
dA_2(t)/dt = a_1*A_1(t) - a_2*A_2(t)
dA_3(t)/dt = a_2*A_2(t) - r_c*A_3(t)

In the simulatin, the codes solve the above protein turnover dynamics for n=1,2,3.

In the simulation, a_i = k_i*B, i=0,1,...,n-1
where k_i denotes the protein's (i+1)-th modification rate coefficient and
B denotes the ubiquitin ligase or kinase concentration.

- A(t), A_i(t) (i=0,...,n), and B are in the unit of nM,
- g(t) is in the unit of nM hour^-1,
- r_c is in the unit of hour^-1,
- k_i (i=0,...,n-1) are in the unit of nM^-1 hour^-1.

The form of g(t) is given as follows:
g(t) = g_max{1 - alpha_g/2*[1 + cos(2*pi/T*t)]}
- g_max   : the peak level of g(t) (unit = nM hour^-1),
- alpha_g : the peak-to-trough difference of g(t),
- T       : the oscillation period of g(t) (in the simultion, T = 24 hour).


*** Main module
The main modules of the codes are
"solve_rhythmic_degradation_g_n1", "solve_rhythmic_degradation_g_n2",
and "solve_rhythmic_degradation_g_n3" in
"e_rhythmic_degradation_n1.py", "e_rhythmic_degradation_n2.py",
and "e_rhythmic_degradation_n3.py", respectively.

"solve_rhythmic_degradation_g_t_n1" module requires the following arguments:
t_start,t_end,dt,T_period,g_max,alpha_g,B_eval,r_c,k_0

"solve_rhythmic_degradation_g_t_n2" module requires the following arguments:
t_start,t_end,dt,T_period,g_max,alpha_g,B_eval,r_c,k_1,k_0

"solve_rhythmic_degradation_g_t_n3" module requires the following arguments:
t_start,t_end,dt,T_period,g_max,alpha_g,B_eval,r_c,k_2,k_1,k_0

- t_start  : the start time of the simulation,
- t_end    : the end time of the simulation,
- dt       : the time step size of the simulation,
- T_period : the the oscillation period of g(t) (in this simulation T=24 hour),
- g_max    : the peak level of g(t) (unit = nM hour^-1),
- alpha_g  : the peak-to-trough difference of g(t),
- B_eval   : the concentraion of proteolytic mediator of protein A (unit = nM),
- r_c      : the substrate turnover rate upon n phosphorylation events (unit = hour^-1),
- k_i      : the protein's (i+1)-th modification rate coefficient
             (i=0,...,n-1 and unit = nM^-1 hour^-1).

"solve_rhythmic_degradation_g_t_n*" (* = 1, 2, or 3) compute
- the time points at which to store the computed solutions,
- the concentrations of A(t), A_i(t) (i=1,...,n) at the time points,
- the protein degradation rate, r(t), r_QSSA(t) = r_c*A_n_QSSA(t)/A(t),
r_gamma(t) = r_c*A_n_gamma(t)/A(t), and R(t) = -A'(t)/A(t) at the time points.

For the use of this module, "solve_rhythmic_degradation_g_t_n*" (* = 1, 2, or 3)
please refer "main_E_rhythmic_degradation_n*" (* = 1, 2, or 3) module in the code.


*** Demos
We provide the example module "main_E_rhythmic_degradation_n*" (* = 1, 2, or 3)
for running "solve_rhythmic_degradation_g_t_n*" (* = 1, 2, or 3). The modules
"main_E_rhythmic_degradation_n*" (* = 1, 2, or 3) are the numerical simulations
for rhythmic degradation of proteins. The kinetic parameters for
the simulations are given in "main_E_rhythmic_degradation_n*" (* = 1, 2, or 3).
The simulation time interval is (0 hour,720 hour) with a uniform time step size 0.05 hour.
Running "main_E_rhythmic_degradation_n*" (* = 1, 2, or 3) with the given
kinetic parameters and the settings takes less than 1 minute on a "normal desktop".


*** Contact information
This code was written by Roktaek Lim. If you have any question,
please contact R. Lim via "rokt.lim@gmail.com".
