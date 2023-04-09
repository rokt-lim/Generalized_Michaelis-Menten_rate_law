"""
This file contains the ODEs for the following models. 

The prefixes of the methods conatin the following meanings. 
PAR : Positive Auto Regulation Model
NFL : No Regulation Model
NAR : Negative Auto Regulation Model

The postfixes of the methods conatin the following meanings. 
_F      : Full Equation version, modified version for Quasi-steady-state approximation.
_FE     : Full Eqaution version, exact version. For the NFL model, there is no approximation, 
            therefore there is no "_FE" version for the NFL model. 
_S      : Model versions where the complex amount is caculated with the sQSSA. 
_T      : Model versions where the complex amount is caculated with the QSSA. 
_T_mass : Model versions where the complex amount is caculated with the QSSA where 
            the protein-protein interaction is considered as the mass action equation model. 
_R      : Model versions where the complex amount is caculated with the revised Michaelis-Menten 
            equation. For "_R" version, there are two additional methods of which the postfix is 
            "_R_Y2", "_R_C_TF_r". The one with "_R_Y2" calculates and records the activator dimer 
            amount and the one with "R_C_TF_r" calculates and records the complex of the activator
            dimer and the DNA. 
_R_mass : Model versions where the complex amount is caculated with revised Michaelis-Menten equation
            with the protein protein interaction is considered as the mass action equation. For "_R_mass" 
            version, there are two additional methods of which the postfix is "_R_Y2_mass", "_R_C_TF_r_mass". 
            The methods are dummy methods which are made for to feed "ddeint" module. 

For example, the method "PAR_F" means the Positive Autoregulation Model in the Full Equation model version. 

In this Python file, the variable "nu" represents the inducer activity, whereas in the main manuscript, the 
symbol "eta" represents the same quantity. 

Parameter variables ("P", "D_TF", "sigma", and so on.) other than "nu" in this Python file is same as in the main manuscript. 

Other variables contains the following meanings. 
CTF : The concentration of the complex of activator dimer and the DNA.  
M   : The concentration of the mRNA. 
A   : The total concentration of the activator monomer. 
A2  : The total concentration of the activator dimer. 

This Python file requires the Numpy library. 

This Python code is written by Junghun Chae.

If you have any questions, please email to the following address:
junghun98@unist.ac.kr

For more detailed information please read Readme.txt
"""
import numpy as np

def PAR_F(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    
    CTF = y[0]
    M   = y[1]
    A   = y[2]
    A2  = y[3]
    
    dCTF = P*nu*(1-CTF)*A2 - D_TF*CTF
    dM  = (1-sigma)*CTF + sigma - B0*M
    dA  = M - A 
    dA2 =  R*(A-2*A2)**2 - D*A2 
    
    return [dCTF, dM, dA, dA2]
#

def PAR_FE(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    
    kd      = para_dict["k_d"]
    rc      = para_dict["rc"]
    kdlt   = para_dict["k_dlt"]
    
    coef_A  = para_dict["coef_A"]
    coef_CTF= para_dict["coef_CTF"]
    
    CTF = y[0]
    M   = y[1]
    A   = y[2]
    A2  = y[3]
    
    dCTF = P*nu*(1-CTF)*A2 - D_TF*CTF
    dM  = (1-sigma)*CTF + sigma -B0*M
    dA  = M - rc / (rc + kdlt) * (A - 2*coef_A/coef_CTF*CTF) - kdlt / (rc + kdlt) * A
    dA2 = R*(A - 2*A2)**2 - (rc + kd) / (rc + kdlt) *(A2 - coef_A/coef_CTF*CTF) - kdlt / (rc + kdlt)*A2
    
    return [dCTF, dM, dA, dA2]
#

def PAR_S(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    kappa   = D/(4*R) 
    M   = y[1]
    A   = y[2]
    
    A2  = A**2 / (4 * (A + kappa))
    CTF = P*nu*A2/(P*nu*A2 + D_TF)
    
    dCTF = 0
    dM  = (1-sigma)*CTF + sigma -B0*M
    dA  = M - A 
    dA2 =  0
    
    return [dCTF, dM, dA, dA2]
#

def PAR_T(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    kappa   = D/(4*R) 
    M   = y[1]
    A   = y[2]
    
    A2  = 1/2 * ( A + kappa - np.sqrt(kappa * (2*A + kappa)) ) 
    CTF = P*nu*A2/(P*nu*A2 + D_TF)
    
    dCTF = 0
    dM  = (1-sigma)*CTF + sigma - B0*M
    dA  = M - A 
    dA2 =  0
    
    return [dCTF, dM, dA, dA2]
#

def PAR_T_mass(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    kappa   = D/(4*R) 
    M   = y[1]
    A   = y[2]
    
    A2  = 1/2 * ( A + kappa - np.sqrt(kappa * (2*A + kappa)) ) 
    CTF = P*nu*A2/(P*nu*A2 + D_TF)
    
    dCTF = 0
    dM  = 0
    dA  = (1-sigma)/B0 *(P*nu/(4*kappa*D_TF) * A**2) / (P*nu/(4*kappa*D_TF)*A**2  + 1) + sigma/B0 - A
    dA2 =  0
    
    return [dCTF, dM, dA, dA2]

def PAR_R(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    
    M   = y(t)[0]
    A   = y(t)[1]
    
    A_2_r = Y2(t)[0] 
    time_shift_tf = 1 / (D_TF + P*nu*A_2_r) 
    
    
    t_tf = t - time_shift_tf
    
    A_2_r_shifted = Y2(t_tf)[0]
    C_TF_r = P*nu*A_2_r_shifted/(P*nu*A_2_r_shifted + D_TF)
    
    dM = ((((1-sigma) * C_TF_r + sigma))/B0 - M )*B0
    dA = M - A 

    return [dM, dA]

def PAR_R_Y2(t, y, Y2, para_dict):
    R       = para_dict["R"]
    kappa   = para_dict["kappa"]
    
    A   = y(t)[1]
    
    delta_tQ = kappa*(kappa+2*A)
    
    time_shift_ppi = 1/(4*R*np.sqrt(delta_tQ))
    
    t_ppi = t - time_shift_ppi
    
    A_shifted = y(t_ppi)[1]
    delta_tQ_shifted = kappa*(kappa + 2 * A_shifted)
    A_2_tQ_shifted = 1/2*(A_shifted + kappa - np.sqrt(delta_tQ_shifted))
    A_2_r = min(1/2*A, A_2_tQ_shifted)
    
    return A_2_r

def PAR_R_C_TF_r(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    
    A_2_r = Y2(t)[0]
    time_shift_tf = 1 / (D_TF + P * nu*A_2_r) 
    
    t_tf = t - time_shift_tf
    
    A_2_r_shifted = Y2(t_tf)[0]
    C_TF_r = P*nu*A_2_r_shifted/(P*nu*A_2_r_shifted + D_TF)
    
    return C_TF_r

def PAR_R_mass(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    D       = para_dict["D"]
    R       = para_dict["R"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    kappa   = D/(4*R) 

    A   = y(t)[1]
    
    tau_shift_ppi   = 1/(4*R*kappa) 
    A_ppi           = y(max(t-tau_shift_ppi,0))[1]
    tau_shift_tf    = 1/D_TF/(1+P*nu*A_ppi**2/(4*kappa*D_TF))
    A_tf            = y(max(t-tau_shift_tf,0))[1]
    
    dA = (1-sigma)/B0 * (P*nu/(4*kappa*D_TF)*A_tf**2) / (P*nu/(4*kappa*D_TF)*A_tf**2 + 1) + sigma/B0 - A
    return [0, dA]

def PAR_R_Y2_mass(t, y, Y2, para_dict):
    return 0

def PAR_R_C_TF_r_mass(t, y, Y2, para_dict):
    return 0
#

def  NFL_F(t, y, para_dict):
    nu      = para_dict["nu"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    
    M   = y[1]
    A   = y[2]
    A2  = y[3]
    
    dCTF = 0
    dM  = (nu + sigma)/(nu + 1) - B0*M
    dA  = M - A 
    dA2 = 0
    
    return [dCTF, dM, dA, dA2]
#

def NAR_F(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    
    CTF = y[0]
    M   = y[1]
    A   = y[2]
    A2  = y[3]
    
    dCTF = P/nu*(1-CTF)*A2 - D_TF*CTF
    dM  = (1-sigma)*(1-CTF) + sigma -B0*M
    dA  = M - A 
    dA2 = R*(A - 2*A2)**2 - D*A2
    
    return [dCTF, dM, dA, dA2]
#

def NAR_FE(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    
    kd      = para_dict["k_d"]
    rc      = para_dict["rc"]
    kdlt    = para_dict["k_dlt"]
    
    coef_A  = para_dict["coef_A"]
    coef_CTF= para_dict["coef_CTF"]
    
    CTF = y[0]
    M   = y[1]
    A   = y[2]
    A2  = y[3]
    
    dCTF = P/nu*(1-CTF)*A2 - D_TF*CTF
    dM  = (1-sigma)*(1-CTF) + sigma -B0*M
    dA  = M - rc / (rc + kdlt) * (A - 2*coef_A/coef_CTF*CTF) - kdlt / (rc + kdlt) * A
    dA2 = R*(A - 2*A2)**2 - (rc + kd) / (rc + kdlt) *(A2 - coef_A/coef_CTF*CTF) - kdlt / (rc + kdlt)*A2
    
    return [dCTF, dM, dA, dA2]
#

def NAR_T(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    kappa   = D/(4*R) 
    M   = y[1]
    A   = y[2]
    
    A2  = 1/2 * ( A + kappa - np.sqrt(kappa * (2*A + kappa)) ) 
    CTF = P*A2/(P*A2 + nu*D_TF)
    
    dCTF = 0
    dM  = (1-sigma)*(1-CTF) + sigma -B0*M
    dA  = M - A 
    dA2 =  0
    
    return [dCTF, dM, dA, dA2]
#

def NAR_S(t, y, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    R       = para_dict["R"]
    D       = para_dict["D"]
    kappa   = D/(4*R) 
    
    M   = y[1]
    A   = y[2]
    
    A2  = A**2 / (4 * (A + kappa))
    CTF = P*A2/(P*A2 + nu*D_TF)
    
    dCTF = 0
    dM  = (1-sigma)*(1-CTF) + sigma -B0*M
    dA  = M - A 
    dA2 =  0
    
    return [dCTF, dM, dA, dA2]
#

def NAR_R(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]

    M   = y(t)[0]
    A   = y(t)[1]
    
    A_2_r = Y2(t)[0]
    time_shift_tf = nu / (nu*D_TF + P *A_2_r) 
    
    t_tf = max(t - time_shift_tf, 0)
    
    A_2_r_shifted = Y2(t_tf)[0]
    C_TF_r = P*A_2_r_shifted/(P*A_2_r_shifted + nu*D_TF)
    
    dM = (1-sigma) * (1-C_TF_r) + sigma - B0 * M 
    dA = M - A 
    
    return [dM, dA]


def NAR_R_Y2(t, y, Y2, para_dict):
    R       = para_dict["R"]
    kappa   = para_dict["kappa"]
    
    A   = y(t)[1]
    
    delta_tQ = kappa*(kappa+2*A)
    
    time_shift_ppi = 1/(4*R*np.sqrt(delta_tQ))
    t_ppi = max(t - time_shift_ppi, 0)
    
    A_shifted = y(t_ppi)[1]
    delta_tQ_shifted = kappa*(kappa + 2 * A_shifted)
    A_2_tQ_shifted = 1/2*(A_shifted + kappa - np.sqrt(delta_tQ_shifted))
    A_2_r = min(1/2*A, A_2_tQ_shifted)
    
    return A_2_r

def NAR_R_C_TF_r(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]

    A_2_r = Y2(t)[0]
    time_shift_tf = nu / (nu*D_TF + P *A_2_r) 
    
    t_tf = max(t - time_shift_tf, 0)
    A_2_r_shifted = Y2(t_tf)[0]
    C_TF_r = P*A_2_r_shifted/(P*A_2_r_shifted + nu*D_TF)
    
    return C_TF_r
#

def NAR_R_mass(t, y, Y2, para_dict):
    P       = para_dict["P"]
    nu      = para_dict["nu"]
    D_TF    = para_dict["D_TF"]
    D       = para_dict["D"]
    R       = para_dict["R"]
    sigma   = para_dict["sigma"]
    B0      = para_dict["B0"]
    kappa   = D/(4*R) 

    A   = y(t)[1]
    
    tau_shift_ppi   = 1/(4*R*kappa) 
    A_ppi           = y(t-tau_shift_ppi)[1]
    tau_shift_tf    = 1/D_TF/(1+P/(4*kappa*nu*D_TF)*A_ppi**2) + 1/D
    A_tf            = y(t-tau_shift_tf) [1]
    
    dA = (1-sigma)/B0 / (P/4/kappa/D_TF/nu *A_tf**2  + 1) + sigma/B0 - A
    
    return [0, dA]


def NAR_R_Y2_mass(t, y, Y2, para_dict):
    return 0

def NAR_R_C_TF_r_mass(t, y, Y2, para_dict):
    return 0
#


