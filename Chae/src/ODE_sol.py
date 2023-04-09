"""
This file contains methods to calculate the epsilon values for the revised michaelis menten equations and the tQSSA equations. 

This Python file requires the Numpy and Pandas library. 
This Python code is written by Junghun Chae.

If you have any questions, please email to the following address:
junghun98@unist.ac.kr

For more detailed information please read Readme.txt
"""

import math 
import os

import numpy as np 
import pandas as pd

import src.parameters as param
import src.epsilons as eps

def make_dir(file_path):
    try : 
        os.makedirs(file_path)
    except : 
        pass
    #
#

class ODE_sol:
    def __init__(self, sol, model_name, params : param.parameters, y0, td=0, solve_module = "scipy", save_dir = "./result/", plot_dir = "plot/") -> None:
        self.y_initial = y0
        self.model_name = model_name
        self.params = params
        self.param_id = params.param_id
        
        self.eps_max_dict = {
            "max_ep_r"        :None,
            "max_ep_QSSA"      :None,
            "mean_ep_r"        :None,
            "mean_ep_QSSA"      :None,
        } 

        self.ss = None
        self.save_dir = save_dir
        self.plot_dir = self.save_dir + plot_dir 
        make_dir(save_dir)
        make_dir(self.plot_dir)
        
        if solve_module == "scipy":
            self.vari_dict = {
                "tau_eval"  : sol.t,
                "C_TF_bbar" : sol.y[0],
                "M_bar"     : sol.y[1],
                "A_bar"     : sol.y[2], 
                "A_2_bar"   : sol.y[3],
                "A_TF_bar"  : sol.y[3],
                "tau_time_delay" : np.ones(len(sol.t)),
                "tau_epsilon"    : np.ones(len(sol.t)),
            }
            
        elif solve_module == "ddeint" : 
            self.vari_dict = {
                "tau_eval"    : sol["tau_eval"],
                "C_TF_r_bbar" : sol["C_TF_r_bbar"],
                "M_bar"       : sol["M_bar"],
                "A_bar"       : sol["A_bar"], 
                "A_2_r_bar"   : sol["A_2_r_bar"],
                "A_TF_bar"    : sol["A_2_r_bar"],
                "tau_epsilon"    : np.ones(len(sol["tau_eval"]))
            }
        
        self.time_delay = td
        self.multiple = 1.5
        dt = self.vari_dict["tau_eval"][-1] - self.vari_dict["tau_eval"][-2]
        self.time_delay_idx = int(self.time_delay/dt)
        self.vari_dict["tau_epsilon"][:int(self.time_delay_idx * self.multiple)] = False
        
        self.update_delta_tQ()
        self.update_A_2_tQ_bar()
        
        self.update_ss()
        self.time_ss_idx = int(self.time_ss/dt)
        
        if "_F" in model_name: 
            self.update_response_time()
        #
        if model_name == "PAR_T": 
            self.vari_dict["A_TF_bar"] = self.vari_dict["A_2_tQ_bar"]
            self.update_C_TF_Q_bbar_p()
            self.calculate_epsilons_T_p()
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict["tau_epsilon"][:int(self.time_delay_idx * self.multiple)] = False
            self.update_ss()
            self.time_ss_idx = int(self.time_ss/dt)
            self.update_response_time()
        #
        if model_name == "NAR_T": 
            self.vari_dict["A_TF_bar"] = self.vari_dict["A_2_tQ_bar"]
            self.update_C_TF_Q_bbar_n()
            self.calculate_epsilons_T_n()
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict["tau_epsilon"][:int(self.time_delay_idx * self.multiple)] = False
            self.update_ss()
            self.time_ss_idx = int(self.time_ss/dt)
            self.update_response_time()
        #   
        if "_S" in model_name: 
            self.update_response_time()
        #
        if model_name == "PAR_R":
            self.vari_dict["C_TF_r_bbar_p"] = self.vari_dict["C_TF_r_bbar"]
            self.update_C_TF_Q_bbar_p()
            self.update_time_delay(A_2_r_bar_initial=self.y_initial[3], C_TF_r_bbar_initial=self.y_initial[0])
            self.update_tau_epsilon()
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict = self.remove_nan_dict(self.vari_dict)
            self.calculate_epsilons_r_p()
            self.update_time_delay(A_2_r_bar_initial=self.y_initial[3], C_TF_r_bbar_initial=self.y_initial[0])
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict = self.remove_nan_dict(self.vari_dict)
            self.update_response_time()

        if model_name == "NAR_R":
            self.vari_dict["C_TF_r_bbar_n"] = self.vari_dict["C_TF_r_bbar"]
            self.update_C_TF_Q_bbar_n()
            self.update_time_delay(A_2_r_bar_initial=self.y_initial[3], C_TF_r_bbar_initial=self.y_initial[0])
            self.update_tau_epsilon()
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict = self.remove_nan_dict(self.vari_dict)
            self.calculate_epsilons_r_n()
            self.update_time_delay(A_2_r_bar_initial=self.y_initial[3], C_TF_r_bbar_initial=self.y_initial[0])
            self.time_delay_idx = int(self.time_delay/dt)
            self.vari_dict = self.remove_nan_dict(self.vari_dict)
            self.update_response_time()
            #
        # 
        
        if "_mass" in model_name: 
            self.update_response_time()
        #
    #

    def update_delta_tQ(self): 
        A_bar = self.vari_dict["A_bar"]
        kappa = self.params.para_dict["kappa"]
        delta_tQ = kappa*(kappa + 2*A_bar)
        
        self.vari_dict.update(
            {
                "delta_tQ" : delta_tQ
            }
        )
    # 

    def update_A_2_tQ_bar(self):
        A_bar = self.vari_dict["A_bar"]
        kappa = self.params.para_dict["kappa"]
        delta_tQ = self.vari_dict["delta_tQ"]
        
        self.vari_dict.update(
            {
                "A_2_tQ_bar" : 1/2*(A_bar + kappa - np.sqrt(delta_tQ))
            }
        ) 
    #
    
    def update_C_TF_Q_bbar_n(self): 
        A_TF_bar  = self.vari_dict["A_TF_bar"]
        P         = self.params.para_dict["P"]
        D_TF      = self.params.para_dict["D_TF"]
        nu        = self.params.para_dict["nu"]
        
        self.vari_dict["C_TF_Q_bbar_n"] = P*A_TF_bar / (P*A_TF_bar+nu*D_TF)
    # 
    
    def update_C_TF_Q_bbar_p(self): 
        A_TF_bar  = self.vari_dict["A_TF_bar"]
        P         = self.params.para_dict["P"]
        D_TF      = self.params.para_dict["D_TF"]
        nu        = self.params.para_dict["nu"]
        
        self.vari_dict["C_TF_Q_bbar_p"] = P*nu*A_TF_bar / (P*nu*A_TF_bar+D_TF)
    # 
    
    def calculate_epsilons_r_p(self): 
        self.eps_dict = {
            "ep_1"      : eps.ep_1(**self.params.para_dict, **self.vari_dict),
            "ep_2"      : eps.ep_2(**self.params.para_dict, **self.vari_dict),
            "ep_r"      : eps.ep_r(**self.params.para_dict, **self.vari_dict),
            "ep_tf"     : eps.ep_tf_p(**self.params.para_dict, **self.vari_dict),
            "ep_tfr"    : eps.ep_tfr_p(**self.params.para_dict, **self.vari_dict),
            "ep_r_PPI": eps.ep_r_PPI(**self.params.para_dict, **self.vari_dict),
            "ep_r_TF" : eps.ep_r_TF_p(**self.params.para_dict, **self.vari_dict),
            "ep_r"    : eps.ep_r_p(**self.params.para_dict, **self.vari_dict)
        } 
        self.eps_dict = self.remove_nan_dict(self.eps_dict)
        self.eps_dict["tau_eval"] = self.vari_dict["tau_eval"]
        
        for key, item in self.eps_dict.items():
            if len(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx]) == 0 :
                self.eps_max_dict["max_" + key] = 0
                self.eps_max_dict["mean_" + key] = 0
            else: 
                self.eps_max_dict["max_" + key] = np.max(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
                self.eps_max_dict["mean_" + key] = np.mean(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
        #
    # 
    
    def calculate_epsilons_r_n(self): 
        self.eps_dict = {
            "ep_1"      : eps.ep_1(**self.params.para_dict, **self.vari_dict),
            "ep_2"      : eps.ep_2(**self.params.para_dict, **self.vari_dict),
            "ep_r"      : eps.ep_r(**self.params.para_dict, **self.vari_dict),
            "ep_tf"     : eps.ep_tf_n(**self.params.para_dict, **self.vari_dict),
            "ep_tfr"    : eps.ep_tfr_n(**self.params.para_dict, **self.vari_dict),
            "ep_r_PPI": eps.ep_r_PPI(**self.params.para_dict, **self.vari_dict),
            "ep_r_TF" : eps.ep_r_TF_n(**self.params.para_dict, **self.vari_dict),
            "ep_r"    : eps.ep_r_n(**self.params.para_dict, **self.vari_dict)
        } 
        self.eps_dict = self.remove_nan_dict(self.eps_dict)
        self.eps_dict["tau_eval"] = self.vari_dict["tau_eval"]
        
        for key, item in self.eps_dict.items():
            if len(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx]) == 0 :
                self.eps_max_dict["max_" + key] = 0
                self.eps_max_dict["mean_" + key] = 0
            else: 
                self.eps_max_dict["max_" + key] = np.max(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
                self.eps_max_dict["mean_" + key] = np.mean(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
        #
    # 
    
    def calculate_epsilons_T_p(self):
        self.eps_dict = {
            "ep_1"      : eps.ep_1(**self.params.para_dict, **self.vari_dict),
            "ep_2"      : eps.ep_2(**self.params.para_dict, **self.vari_dict),
            "ep_tq"     : eps.ep_tq(**self.params.para_dict, **self.vari_dict),
            "ep_tf"     : eps.ep_tf_p(**self.params.para_dict, **self.vari_dict),
            "ep_tfq"    : eps.ep_tfq_p(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA_PPI" : eps.ep_QSSA_PPI(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA_TF"  : eps.ep_QSSA_TF_p(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA"     : eps.ep_QSSA_p(**self.params.para_dict, **self.vari_dict)
        } 
        
        self.eps_dict = self.remove_nan_dict(self.eps_dict)
        self.eps_dict["tau_eval"] = self.vari_dict["tau_eval"]
        
        for key, item in self.eps_dict.items():
            if len(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx]) == 0 :
                self.eps_max_dict["max_" + key] = 0
                self.eps_max_dict["mean_" + key] = 0
            else: 
                self.eps_max_dict["max_" + key] = np.max(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
                self.eps_max_dict["mean_" + key] = np.mean(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
        #
    # 
    
    def calculate_epsilons_T_n(self):
        self.eps_dict = {
            "ep_1"      : eps.ep_1(**self.params.para_dict, **self.vari_dict),
            "ep_2"      : eps.ep_2(**self.params.para_dict, **self.vari_dict),
            "ep_tq"     : eps.ep_tq(**self.params.para_dict, **self.vari_dict),
            "ep_tf"     : eps.ep_tf_n(**self.params.para_dict, **self.vari_dict),
            "ep_tfq"    : eps.ep_tfq_n(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA_PPI" : eps.ep_QSSA_PPI(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA_TF"  : eps.ep_QSSA_TF_n(**self.params.para_dict, **self.vari_dict),
            "ep_QSSA"     : eps.ep_QSSA_n(**self.params.para_dict, **self.vari_dict)
        } 
        
        self.eps_dict = self.remove_nan_dict(self.eps_dict)
        self.eps_dict["tau_eval"] = self.vari_dict["tau_eval"]
        
        
        for key, item in self.eps_dict.items():
            if len(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx]) == 0 :
                self.eps_max_dict["max_" + key] = 0
                self.eps_max_dict["mean_" + key] = 0
            else: 
                self.eps_max_dict["max_" + key] = np.max(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
                self.eps_max_dict["mean_" + key] = np.mean(item[int(self.time_delay_idx*self.multiple): self.time_ss_idx])
        #
    # 
    
    def save_info(self, fname = "result", save_variables=False, **dfkwargs): 
        fname_sum = self.save_dir + fname + "_summary.csv"
        fname_var = self.save_dir + "var/" + fname + f"_var_{self.model_name}_{self.param_id}.csv"
        make_dir(self.save_dir + "var/")
        
        name_df  = pd.DataFrame({"model_name" : self.model_name}, index = [self.param_id])
        state_df = pd.DataFrame({
                "response_time" : self.response_time
            }, index = [self.param_id])
        param_df = pd.DataFrame(self.params.para_dict, index = [self.param_id])
        
        eps_save_dict = {
            "max_ep_r"         :None,
            "max_ep_QSSA"       :None,
            "mean_ep_r"        :None,
            "mean_ep_QSSA"      :None,
        } 
        
        for key_ep in eps_save_dict.keys():
            eps_save_dict[key_ep] = self.eps_max_dict[key_ep]
        #
        
        eps_max_df = pd.DataFrame(eps_save_dict, index = [self.param_id])
        eps_max_df = eps_max_df[sorted(eps_max_df.columns)]
        
        save_df = pd.concat([name_df, state_df, eps_max_df, param_df], axis = 1)
        save_df.index.name = "param_id" 
        save_df.to_csv(fname_sum, **dfkwargs)
        
        if save_variables: 
            var_df = pd.DataFrame(self.vari_dict)
            var_df.set_index("tau_eval", inplace=True)
            var_df.to_csv(fname_var)
        #     
    # 
    
    def get_eps_max_df(self):
        eps_max_df = pd.DataFrame(self.eps_max_dict, index = [self.param_id])
        eps_max_df.set_index("tau_eval", inplace=True)
        return eps_max_df
    # 
    
    def get_eps_df(self):
        eps_df   = pd.DataFrame(self.eps_dict)
        eps_df.set_index("tau_eval", inplace=True)
        return eps_df
    #
    
    def get_C_TF_S(self):
        P       = self.vari_dict["P"]
        D_TF    = self.vari_dict["D_TF"]
        nu      = self.vari_dict["nu"]
        A_bar   = self.vari_dict["A_bar"]
        kappa   = self.params.para_dict["kappa"]
        
        A_2_bar = A_bar*A_bar / (4 * (A_bar + kappa))
        
        return P*A_2_bar / (P*A_2_bar + nu * D_TF)
    #

    def update_time_delay(self, A_2_r_bar_initial, C_TF_r_bbar_initial): 
        self.vari_dict["tau_time_delay"] = (self.vari_dict["A_2_r_bar"] != A_2_r_bar_initial ) * (self.vari_dict["C_TF_r_bbar"] != C_TF_r_bbar_initial)
        
        D_TF = self.params.para_dict["D_TF"]
        P = self.params.para_dict["P"]
        nu = self.params.para_dict["nu"]
        R = self.params.para_dict["R"]
        kappa = self.params.para_dict["kappa"]
        A_2_r = self.vari_dict["A_2_r_bar"]
        A = self.vari_dict["A_bar"]
        
        delta_tQ = kappa*(kappa+2*A)
        
        if "PAR" in self.model_name:
            tf_delay = D_TF / (D_TF + P * nu*A_2_r) 
            ppi_delay = 1/(4*R*np.sqrt(delta_tQ))
        #
        if "NAR" in self.model_name:
            tf_delay = nu*D_TF / (nu*D_TF + P *A_2_r) 
            ppi_delay = 1/(4*R*np.sqrt(delta_tQ))
        #
        if self.time_ss_idx == 0 : 
            self.eps_max_dict["time_delay_max"] = 0
            self.eps_max_dict["time_delay_mean"] = 0
        else :     
            self.eps_max_dict["time_delay_max"] = np.max(np.array(tf_delay[:self.time_ss_idx]+ppi_delay[:self.time_ss_idx]))
            self.eps_max_dict["time_delay_mean"] = np.mean(np.array(tf_delay[:self.time_ss_idx]+ppi_delay[:self.time_ss_idx]))
        # 
    #
    
    def update_tau_epsilon(self): 
        idx_dispose = math.ceil( self.get_time_delay_idx() * self.multiple ) 
        tau_ep = self.vari_dict["tau_time_delay"] 
        tau_ep[:idx_dispose + 1] = False
        
        self.vari_dict["tau_epsilon"] = tau_ep
    #
    
    def get_time_delay_idx(self): 
        idx_list = np.where( self.vari_dict["tau_time_delay"] == False)[0] 
        if len(idx_list) == 0 :
            self.time_delay  = self.vari_dict["tau_eval"] [0]
            return 0 
        if len(idx_list) == 1 : 
            self.time_delay  = self.vari_dict["tau_eval"] [idx_list[0]]
            return idx_list[0]
        
        physical_max_idx = max(idx_list)
        self.time_delay  = self.vari_dict["tau_eval"] [physical_max_idx]
        return physical_max_idx
    #
    
    def remove_nan_dict(self, dict_t, var=0):
        for key, item in dict_t.items():
            dict_t[key] = np.nan_to_num(item, nan=var)
        #
        return dict_t
    #
    
    def update_ss(self, thr=0.01): 
        
        dtau_idx = int( 1 / (self.vari_dict["tau_eval"][-1] - self.vari_dict["tau_eval"][-2]) )
        
        A_bar = self.vari_dict["A_bar"]
        
        reach_ss, t_idx, ss = self.get_ss_idx(A_bar, dtau_idx)
        self.reach_ss = reach_ss 
        self.time_ss  = self.vari_dict["tau_eval"][t_idx]
        
        self.ss_dict = {}
        non_vari_list = ["tau_time_delay", "tau_epsilon", "tau_eval", "delta_tQ"]
        for key, item in self.vari_dict.items(): 
            
            if key in non_vari_list: 
                continue
            #            
            reach_ss, t_idx, ss = self.get_ss_idx(item, dtau_idx)
            if reach_ss:
                self.ss_dict[key] = ss
            else : 
                self.ss_dict[key] = -1 
            #
        #
    #
    
    @staticmethod
    def get_ss_idx(arr, dtau_idx, thr=0.01):
    
        for a_idx in range(len(arr) - 2 * dtau_idx):
            v1 = arr[a_idx]
            v2 = arr[a_idx+dtau_idx]
            v3 = arr[a_idx+dtau_idx*2]
            
            dv1 = abs( v2 - v1 )
            dv2 = abs( v3 - v2 )
            m = (v1 + v2 + v3)/3
            
            
            if min(v1, v2, v3) == 0 : 
                continue
            
            if (max(v1, v2, v3)/min(v1, v2, v3) < 1 + thr) and (dv1 > dv2):
                
                return (1, a_idx, m)
            #
        #
        return (0, -1, -1)
    #
    
    def update_response_time(self): 
        if not self.reach_ss: 
            self.response_time = -1 
            return 0
        # 
        
        self.ss = self.ss_dict["A_bar"]
        
        half_ss = (self.y_initial[2] + self.ss) /2  
        
        if self.y_initial[2] > self.ss:
            dif = - self.vari_dict["A_bar"] + half_ss 
        else:
            dif = self.vari_dict["A_bar"] - half_ss 
        dif = dif > 0
        dif = np.where(dif == True)[0]
        rt_idx = min(dif)
        
        self.response_time = self.vari_dict["tau_eval"][rt_idx]
    # 
#