Manual for the codes
This document describes the information about the following files: 
1. "src/ddeint_modified.py”
2. "src/epsilons.py"
3. "src/model_ODEs.py"
4. "src/ODE_sol.py"
5. "src/parameters.py"

Note: The “src/ddeint_modified.py” is originally from “https://github.com/Zulko/ddeint” and modified by 
J. Chae. The code has the “Public Domain” license.

System Requirements
These Python codes can run on Windows, Linux, or macOS. Python 3.7 or a later version is required. 
It requires the following Python packages. 
-NumPy 1.15.1 or greater
-SciPy 1.3.1 or greater
-Pandas 1.0 or greater

General Information
These codes simulate the simple gene expression model with and without autoregulations. 
There are 3 models. 
1. Positive autoregulation model: This model describes a system where a gene is transcripted 
    and translated to produce proteins, and the proteins form dimers and the dimer promotes
    the activity of the DNA transcription. 
2. Negative autoregulation model: This model describes a system where a gene is transcripted 
    and translated to produce proteins, and the proteins form dimers and the dimer reduces 
    the activity of the DNA transcription. 
3. No feedback loop model: This model describes a constitutive gene expression model. 

In all 3 models, the transcription level is modulated by the inducer.

Protein-protein interaction and TF-DNA binding equations are expressed with the C_γ, QSSA, sQSSA formulas, 
and a full equation without approximations. Parameters are randomly sampled to test the validity of each 
approximation. The validity is tested in terms of the response time which is defined as the time the protein 
concentration takes to achieve 90% concentration of its steady-state level. 

Directories Required
The following directories are required for the codes. 
Directories: 
parameters/
result/

Input Files Required
The following files are required for the codes. 
Files:
parameters/parameter.csv

Data Produced
The parameters are saved to “results/result_summary.csv”. The CSV file has a total of 31 columns.  The meaning of each column
is described below:

case_id		: ID for each parameter set. Each parameter set has one ID.
model_name	: There are 3 models (Positive autoregulation, Negative autoregulation, and No feedback loop) with different approximations. 
            The details of the models are described below: 

            Details of the models. 
            There are 13 versions. 
            "PAR_F", "PAR_FE", "PAR_R", "PAR_R_mass", "PAR_S", "PAR_T", "NAR_F", "NAR_FE", "NAR_R", "NAR_R_mass", "NAR_S", 
            "NAR_T", "NFL".

            The prefixes of the model versions conatain the following meanings. 
            PAR : Positive Auto Regulation Model
            NFL : No Regulation Model
            NAR : Negative Auto Regulation Model

            The postfixes of the model versions contain the following meanings. 
            _F      : Full Equation version, modified version for Quasi-steady-state approximation.
            _FE     : Full Equation version, exact version. For the NFL model, there is no approximation, 
                        therefore there is no "_FE" version for the NFL model. 
            _S      : Model versions where the complex amount is calculated with the sQSSA. 
            _T      : Model versions where the complex amount is calculated with the QSSA. 
            _T_mass : Model versions where the complex amount is calculated with the QSSA where 
                        the protein-protein interaction is considered as the mass action equation model. 
            _R      : Model versions where the complex amount is calculated with the revised Michaelis-Menten 
                        equation. For "_R" version, there are two additional methods of which the postfix is 
                        "_R_Y2", "_R_C_TF_r". The one with "_R_Y2" calculates and records the activator dimer 
                        amount and the one with "R_C_TF_r" calculates and records the complex of the activator
                        dimer and the DNA. 
            _R_mass : Model versions where the complex amount is calculated with revised Michaelis-Menten equation
                        with the protein-protein interaction is considered as the mass action equation. For "_R_mass" 
                        version, there are two additional methods of which the postfix is "_R_Y2_mass", "_R_C_TF_r_mass". 
                        The methods are dummy methods made to feed the "ddeint" module. 

            For example, the method "PAR_F" means the Positive Autoregulation Model in the Full Equation model version. 

response_time   : The response time calculated by the simulation. 
[epsilons]      : There are 4 epsilon values. 
            "max_ep_r"      : maximum value of the epsilons for the revised Michaelis-Menten equation over the time series. 
            "max_ep_QSSA"  : maximum value of the epsilons for the QSSA over the time series. 
            "mean_ep_r"     : time average value of the epsilons for the revised Michaelis-Menten equation over the time series. 
            "mean_ep_QSSA" : time average value of the epsilons for the QSSA over the time series. 
            NOTE: "mean_ep_r", "mean_ep_QSSA" averages over time until the protein concentration reaches to the steady-state
                    concentration.

[parameters]    : The parameter values.

Size of the result file
With 10,000 parameters for all versions, the output file size would be approximately 37MB. 

Parameters.
K_TF            : The dissociation constant of the dimer-DNA complex. 
k_TFd           : The dissociation rate of the dimer-DNA complex.
k_dlt           : The dilution rate due to the cell growth. 
a0              : The maximum transcription rate.  
a1              : The maximum translation rate. 
b0              : The degradation rate of the mRNA.
rc              : The degradation rate of the transcription factor monomer.
K               : The dissociation constant of the dimer complex. 
k_d             : The dissociation rate of the dimer complex. 
V               : The volume of the bacteria. 
nu              : The effect of the inducer. 

From here, all parameters are derived from the parameters above.

coef_M          : (rc + k_dlt)*V/a0
coef_A          : (rc + k_dlt)**2*V/a0/a1
coef_CTF        : V
coef_t          : (rc + k_dlt)

D_TF            : (k_TFd + k_dlt)/(rc + k_dlt)
D               : (k_d + rc + k_dlt)/(rc + k_dlt) 
sigma           : 0.05
R               : k_a * a0 * a1 / (2 * V * (rc + k_dlt)^3)
B0              : (b0 + k_dlt) / (rc + k_dlt) 
kappa           : D / (4*R) 
P               : 2 * R * K * D_TF / (K_TF * D)
L               : (rc + k_dlt)^2 / (a0 * a1)

Connection between symbols in the main manuscript. 
<Symbols in the main manuscript>  : <Symbols in the code> 
eta             : nu
k_TFa           : k_TFd/K_TF 
k_TFa_hat       : k_TFa/nu 
k_TFd           : k_TFd 
k_a             : k_a 
k_d             : k_d 
k_dlt           : k_dlt 
a_0             : a_0 
a_1             : a_1 
s               : a_0 * sigma, Note, sigma = 0.05
b_0             : b_0  
r_c             : r_c 
V               : V 
D               : D 
D_TF            : D_TF 
sigma           : sigma 
R               : R 
P               : P 
L               : L 
B_0             : B0
kappa           : kappa 

Modeul Information
1. "src/ddeint_modified.py”
This module solves the time-delayed ODE systems. 

2. "src/epsilons.py"
This module calculates the epsilon values for QSSA and the revised Michaelis-Menten equations. 

3. "src/model_ODEs.py"
This module contains various model ODEs. 

4. "src/ODE_sol.py"
This module conatains containers for the result of the simulations.

5. "src/parameters.py"
This module reads parameter range information from the "parameters/parameters.csv" file and sample parameters. 

Demos
Running “run_simulation.py” with Intel i7-7700 CPU and 16GM RAM desktop computer takes less than 5 
minutes to simulate 10 parameters. 
Note: The default parameter conditions might be different from the main manuscript simulation conditions. 

Contact Information
The codes are written by Junghun Chae. If you have any questions, please contact J. Chae via “junghun98@unist.ac.kr”.