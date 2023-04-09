"""
This file contains methods to calculate the epsilon values for the revised michaelis menten equations and the QSSA equations. 

This Python file requires the Numpy and SciPy library. 
This Python code is written by Junghun Chae.

If you have any questions, please email to the following address:
junghun98@unist.ac.kr

For more detailed information please read Readme.txt
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


def get_derivative(t, y, n=1): 
    spl = UnivariateSpline(t, y, k=3, s=0) 
    return spl.derivative(n)(t)
#

def mu(t, y):
    return 1/y*get_derivative(t, y)
#

def get_absmax(x):
    return np.max(np.abs(x))
#

def ep_1(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_bar       = kwargs["A_bar"]
    A_2_tQ_bar  = kwargs["A_2_tQ_bar"]
    R           = kwargs["R"]
    D           = kwargs["D"]
    kappa       = kwargs["kappa"]
    
    delta_tQ = 1 + 2 * A_bar / kappa
    
    return  np.abs( get_derivative(tau_eval * D, A_2_tQ_bar / kappa) / delta_tQ) * kwargs["tau_epsilon"]
#

def ep_2( **kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_bar       = kwargs["A_bar"]
    R           = kwargs["R"]
    D           = kwargs["D"]
    kappa       = kwargs["kappa"]

    delta_tQ = 1 + 2 * A_bar / kappa
    
    return np.abs( 2 / delta_tQ**(3/2) * get_derivative(tau_eval * D, A_bar / 2) ) * kwargs["tau_epsilon"] 
#

def ep_r(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_bar       = kwargs["A_bar"]
    A_2_r_bar   = kwargs["A_2_r_bar"]
    A_2_tQ_bar  = kwargs["A_2_tQ_bar"]
    R           = kwargs["R"]
    D           = kwargs["D"]
    kappa       = kwargs["kappa"]
    
    delta_tQ = 1 + 2 * A_bar / kappa
    
    return np.abs( 1/(2*delta_tQ*A_2_r_bar) * get_derivative(tau_eval * D, A_2_tQ_bar, 2) ) * kwargs["tau_epsilon"]
#

def ep_tq(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_bar       = kwargs["A_bar"]
    A_2_tQ_bar  = kwargs["A_2_tQ_bar"]
    R           = kwargs["R"]
    D           = kwargs["D"]
    kappa       = kwargs["kappa"]

    delta_tQ = 1 + 2 * A_bar / kappa
    return np.abs( 1/A_2_tQ_bar * (1/np.sqrt(delta_tQ) * get_derivative(tau_eval * D, A_2_tQ_bar)) 
                - 1/delta_tQ * get_derivative(tau_eval * D, A_2_tQ_bar, 2))* kwargs["tau_epsilon"]
#

def ep_tf_p(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P * nu / D_TF
    
    return np.abs( 1 / (1 + A_TF_bar) * get_derivative(tau_eval * D_TF, A_TF_bar) ) * kwargs["tau_epsilon"] 
#

def ep_tfr_p(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    C_TF_r_bbar = kwargs["C_TF_r_bbar_p"]
    C_TF_Q_bbar = kwargs["C_TF_Q_bbar_p"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P * nu / D_TF
    return np.abs(1/(2 * (1 + A_TF_bar)**2 * C_TF_r_bbar) * get_derivative(tau_eval * D_TF, C_TF_Q_bbar, 2) ) * kwargs["tau_epsilon"]
#

def ep_tfq_p(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    C_TF_Q_bbar = kwargs["C_TF_Q_bbar_p"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P * nu / D_TF
    return np.abs(1/ C_TF_Q_bbar * ( 1 / (A_TF_bar + 1) * get_derivative(tau_eval * D_TF, C_TF_Q_bbar)
                    - 1 / (A_TF_bar + 1)**2 * get_derivative(tau_eval * D_TF, C_TF_Q_bbar) ) ) * kwargs["tau_epsilon"]
#

def ep_tf_n(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P / (D_TF * nu)
    
    return np.abs( 1 / (1 + A_TF_bar) * get_derivative(tau_eval * D_TF, A_TF_bar) ) * kwargs["tau_epsilon"] 
#

def ep_tfr_n(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    C_TF_r_bbar = kwargs["C_TF_r_bbar_n"]
    C_TF_Q_bbar = kwargs["C_TF_Q_bbar_n"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P / (D_TF * nu)
    
    return np.abs(1/(2 * (1 + A_TF_bar)**2 * C_TF_r_bbar) * get_derivative(tau_eval * D_TF, C_TF_Q_bbar, 2) ) * kwargs["tau_epsilon"]
#

def ep_tfq_n(**kwargs):
    tau_eval    = kwargs["tau_eval"]
    A_TF_bar    = kwargs["A_TF_bar"]
    C_TF_Q_bbar = kwargs["C_TF_Q_bbar_n"]
    D_TF        = kwargs["D_TF"]
    P           = kwargs["P"]
    nu          = kwargs["nu"]
    
    A_TF_bar = A_TF_bar * P / (D_TF * nu)
    return np.abs(1/ C_TF_Q_bbar * ( 1 / (A_TF_bar + 1) * get_derivative(tau_eval * D_TF, C_TF_Q_bbar)
                    - 1 / (A_TF_bar + 1)**2 * get_derivative(tau_eval * D_TF, C_TF_Q_bbar) ) ) * kwargs["tau_epsilon"]
#

def ep_r_PPI(**kwargs): 
    return np.maximum.reduce([
        ep_1(**kwargs),
        ep_2(**kwargs),
        ep_r(**kwargs)
    ])
#

def ep_r_TF_p( **kwargs): 
    return np.maximum(
        ep_tf_p(**kwargs),
        ep_tfr_p(**kwargs)
    )
#

def ep_r_TF_n( **kwargs): 
    return np.maximum(
        ep_tf_n(**kwargs),
        ep_tfr_n(**kwargs)
    )
#

def ep_r_p(**kwargs): 
    return np.maximum(
        ep_r_PPI(**kwargs),
        ep_r_TF_p(**kwargs)
    )
#

def ep_r_n(**kwargs): 
    return np.maximum(
        ep_r_PPI(**kwargs),
        ep_r_TF_n(**kwargs)
    )
#

def ep_QSSA_PPI(**kwargs): 
    return np.maximum.reduce([
        ep_1(**kwargs),
        ep_2(**kwargs),
        ep_tq(**kwargs)
    ])
#

def ep_QSSA_TF_p( **kwargs): 
    return np.maximum(
        ep_tf_p(**kwargs),
        ep_tfq_p(**kwargs)
    )
#

def ep_QSSA_TF_n(**kwargs): 
    return np.maximum(
        ep_tf_n(**kwargs),
        ep_tfq_n(**kwargs)
    )
#

def ep_QSSA_p(**kwargs): 
    return np.maximum(
        ep_QSSA_PPI(**kwargs),
        ep_QSSA_TF_p(**kwargs)
    )
#

def ep_QSSA_n(**kwargs): 
    return np.maximum(
        ep_QSSA_PPI(**kwargs),
        ep_QSSA_TF_n(**kwargs)
    )
#

def ep_p_max(**kwargs):
    return np.maximum(
        ep_r_p(**kwargs),
        ep_QSSA_p(**kwargs)
    )
#

def ep_p_all(**kwargs):
    return {
        "ep_1":         ep_1(**kwargs),
        "ep_2":         ep_2(**kwargs),
        "ep_r":         ep_r(**kwargs),
        "ep_tq":        ep_tq(**kwargs),
        "ep_tf_p":      ep_tf_p(**kwargs),
        "ep_tfr_p":     ep_tfr_p(**kwargs),
        "ep_tfq_p":     ep_tfq_p(**kwargs),
        "ep_r_PPI":   ep_r_PPI(**kwargs),
        "ep_r_TF_p":  ep_r_TF_p(**kwargs),
        "ep_r_p":     ep_r_p(**kwargs),
        "ep_QSSA_PPI": ep_QSSA_PPI(**kwargs),
        "ep_QSSA_TF_p":ep_QSSA_TF_p(**kwargs),
        "ep_QSSA_p":   ep_QSSA_p(**kwargs)
    }
#

def ep_n_max(**kwargs):
    return np.maximum(
        ep_r_n(**kwargs),
        ep_QSSA_n(**kwargs)
    )
#

def ep_n_all(**kwargs):
    return {
        "ep_1":         ep_1(**kwargs),
        "ep_2":         ep_2(**kwargs),
        "ep_r":         ep_r(**kwargs),
        "ep_tq":        ep_tq(**kwargs),
        "ep_tf_n":      ep_tf_n(**kwargs),
        "ep_tfr_n":     ep_tfr_n(**kwargs),
        "ep_tfq_n":     ep_tfq_n(**kwargs),
        "ep_r_PPI":   ep_r_PPI(**kwargs),
        "ep_r_TF_n":  ep_r_TF_n(**kwargs),
        "ep_r_n":     ep_r_n(**kwargs),
        "ep_QSSA_PPI": ep_QSSA_PPI(**kwargs),
        "ep_QSSA_TF_n":ep_QSSA_TF_n(**kwargs),
        "ep_QSSA_n":   ep_QSSA_n(**kwargs)
    }
#

