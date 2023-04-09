"""
This file run the simulation of the PAR, NAR, NFL models. 

This Python file requires the Numpy and Pandas library. 
This Python code is written by Junghun Chae.

If you have any questions, please email to the following address:
junghun98@unist.ac.kr

For more detailed information please read Readme.txt
"""


import numpy as np
from scipy.integrate import solve_ivp

from src import model_ODEs as model
from src import parameters as param
from src import ODE_sol as ode_sol
import src.ddeint_modified as dde

import os

np.seterr(divide="ignore")
np.seterr(invalid="ignore")


eq_dict = {
    "ddeint":
        {
            "PAR_R" : (model.PAR_R, model.PAR_R_Y2, model.PAR_R_C_TF_r), 
            "PAR_R_mass" : (model.PAR_R_mass, model.PAR_R_Y2_mass, model.PAR_R_C_TF_r_mass), 
            "NAR_R" : (model.NAR_R, model.NAR_R_Y2, model.NAR_R_C_TF_r),
            "NAR_R_mass" : (model.NAR_R_mass, model.NAR_R_Y2_mass, model.NAR_R_C_TF_r_mass)
        },
    "solve_ivp":
        {
            "PAR_F"  : model.PAR_F,
            "PAR_FE" : model.PAR_FE,
            "PAR_T"  : model.PAR_T, 
            "PAR_S"  : model.PAR_S, 
            "NFL_F"  : model.NFL_F,
            "NAR_F"  : model.NAR_F,
            "NAR_FE" : model.NAR_FE,
            "NAR_T"  : model.NAR_T, 
            "NAR_S"  : model.NAR_S
        }
}

root_dir    = "./"
param_dir   = root_dir + "parameters/parameters.csv"

param_setting = {
    "fname"     : param_dir,
    "k_d_ver"   : 1, 
    "k_TFd_ver" : 1, 
    "K_ver"     : 1, 
    "K_TF_ver"  : 1, 
    "a0_ver"    : 1, 
    "a1_ver"    : 1
}

def run_simulation(num_simulation):
    tau_begin   = 0
    tau_end     = 15
    dtau        = 0.01
    tau_span    = [tau_begin, tau_end]
    tau_eval    = np.arange(tau_begin, tau_end, dtau)

    res_dir = "./result/"

    params = param.parameters(**param_setting)
    for _ in range(num_simulation): 
        params.sample_again()
        y_initial = [0,0,0,0]
        td_nar = 0
        td_par = 0
        solve_module = "ddeint"
        
        for eq_name, (eq1,eq2,eq3) in eq_dict[solve_module].items():
            y_initial_MP = y_initial[1:3]
            y_initial_CTF = y_initial[0]
            y_initial_A2 = y_initial[3]
            
            sol = dde.ddeint(
                lambda y, t, Y2: eq1(t, y, Y2, params.para_dict),
                lambda y, t, Y2: eq2(t, y, Y2, params.para_dict),
                lambda y, t, Y2: eq3(t, y, Y2, params.para_dict),
                lambda t : y_initial_MP, 
                lambda t : [y_initial_A2], 
                lambda t : [y_initial_CTF],
                tau_eval
            )
            
            sol_dict = {
                "tau_eval"  : tau_eval, 
                "C_TF_r_bbar" : sol[:,3],
                "M_bar" : sol[:,0],
                "A_bar" : sol[:,1],
                "A_2_r_bar" : sol[:,2]
            }
            
            if "mass" in eq_name:
                if "NAR" in eq_name: 
                    td = td_nar
                if "PAR" in eq_name: 
                    td = td_par
                # 
                
                res = ode_sol.ODE_sol(sol = sol_dict, model_name = eq_name, params = params, 
                                solve_module=solve_module, y0=y_initial, td = td, save_dir = res_dir)
            else: 
                res = ode_sol.ODE_sol(sol = sol_dict, model_name = eq_name, params = params, 
                                solve_module=solve_module, y0=y_initial)
            
            if not os.path.exists(res_dir + "result_summary.csv"):
                res.save_info(save_variables = False, fname = f"result")
            else : 
                res.save_info(save_variables = False, mode = "a", header=False, fname = f"result")
            #
            
            if "NAR_R" == eq_name:
                td_nar = res.time_delay
            if "PAR_R" == eq_name: 
                td_par = res.time_delay 
        #
        solve_module = "solve_ivp"

        for eq_name, eq_fun in eq_dict[solve_module].items():
            sol = solve_ivp(
                fun     = lambda t, y: eq_fun(t, y, params.para_dict),
                t_span  = tau_span,
                y0      = y_initial,
                t_eval  = tau_eval,
                method  = "LSODA"
            )

            if "PAR" in eq_name: 
                td = td_par
            if "NAR" in eq_name: 
                td = td_nar
            #
            
            res = ode_sol.ODE_sol(sol, eq_name, params, y0=y_initial, td = td)                    
            
            if not os.path.exists(res_dir + "result_summary.csv"):
                res.save_info(save_variables = False, fname = f"result")
            else : 
                res.save_info(save_variables = False, mode = "a", header=False, fname = f"result")
            #
        #
#

run_simulation(10)