"""
This file contains methods to generate parameter values. 

This Python file requires the Numpy, SciPy and Pandas library. 
This Python code is written by Junghun Chae.

If you have any questions, please email to the following address:
junghun98@unist.ac.kr

For more detailed information please read Readme.txt
"""
import random as rd 

from numpy.random import Generator, MT19937
import scipy.stats as stats
import numpy as np 
import pandas as pd 

class parameters: 
    def __init__(self, fname, k_d_ver = 1, k_TFd_ver = 1, K_ver = 1, K_TF_ver = 1, a0_ver = 1, a1_ver = 1 ): 
        assert k_d_ver >= 1 or k_d_ver <= 3, "kd_ver out of range"
        assert k_TFd_ver >= 1 or k_TFd_ver <= 2, "kdtf_ver out of range"
        assert a0_ver == 1 or a0_ver == 3, "a0ver not valid"
        assert a1_ver == 1 or a1_ver == 3, "a1ver not valid"
        assert K_ver == 1 or K_ver == 2, "K_ver not valid"
        assert K_TF_ver == 1 or K_TF_ver == 2, "K_TF_ver not valid"
        
        self.para_df = pd.read_csv(fname, encoding="utf-8")
        self.para_df.set_index("Parameters", inplace=True)
        
        if K_TF_ver == 2:
            self.para_df["KTF"]["min"] = self.para_df["KTF"]["ver2min"]
            self.para_df["KTF"]["max"] = self.para_df["KTF"]["ver2max"]
        #
    
        if k_TFd_ver == 2:
            self.para_df["k_TFd"]["min"] = self.para_df["k_TFd"]["ver2min"]
            self.para_df["k_TFd"]["max"] = self.para_df["k_TFd"]["ver2max"]
        #
    
        if K_ver == 2:
            self.para_df["KTF"]["min"] = self.para_df["KTF"]["ver2min"]
            self.para_df["KTF"]["max"] = self.para_df["KTF"]["ver2max"]
        #
    
        if k_d_ver == 2:
            self.para_df["k_d"]["min"] = self.para_df["k_d"]["ver2min"]
            self.para_df["k_d"]["max"] = self.para_df["k_d"]["ver2max"]
        #
        elif k_d_ver == 3:
            self.para_df["k_d"]["min"] = self.para_df["k_d"]["ver3"]
            self.para_df["k_d"]["max"] = self.para_df["k_d"]["ver3"]
            self.para_df["k_d"]["Scaling"] = "Fix"
        #
    
        if a0_ver == 3:
            self.para_df["a0"]["min"] = self.para_df["a0"]["ver3"]
            self.para_df["a0"]["max"] = self.para_df["a0"]["ver3"]
            self.para_df["a0"]["Scaling"] = "Fix"
        #
    
        if a1_ver == 3:
            self.para_df["a1"]["min"] = self.para_df["a1"]["ver3"]
            self.para_df["a1"]["max"] = self.para_df["a1"]["ver3"]
            self.para_df["a1"]["Scaling"] = "Fix"
                #
            #
        #
        
        self.param_id = "{:07d}".format(rd.randint(0,10000000))
        self.MT = Generator(MT19937(int(self.param_id)))
        self.gen_dict = {
            "Lin" : {},
            "Log" : {}
        }
        
        self.hard_const = False 
        while not self.hard_const:
            self.sample_parameters()
            self.to_dimensionless()
            self.calculate_ss()
            self.hard_const = self.hard_constraints()
        # 
    # 
    
    def sample_parameters(self):
        self.para_dict = {}
        for idx in self.para_df.index : 
            if self.para_df.loc[idx,"Scaling"] == "Fix": 
                para_value = self.para_df.loc[idx,"min"] 
                assert self.para_df.loc[idx,"min"] == self.para_df.loc[idx,"max"], f"Parameter sampling {idx} with \"Fixed\" mode with different minimum and maximum value"
            else : 
                para_min = self.para_df.loc[idx,"min"]
                para_max = self.para_df.loc[idx,"max"]
                scaling_method = self.para_df.loc[idx,"Scaling"]
                
                if scaling_method == "Lin":    
                    if f"{para_min}{para_max}" not in self.gen_dict["Lin"].keys(): 
                        scipy_gen = stats.uniform(loc = para_min, scale = para_max - para_min)
                        scipy_gen.random_state = self.MT 
                        self.gen_dict["Lin"][f"{para_min}{para_max}" ] = scipy_gen
                    #
                    para_value = self.gen_dict["Lin"][f"{para_min}{para_max}" ].rvs()
                if scaling_method == "Log":
                    if f"{para_min}{para_max}" not in self.gen_dict["Log"].keys(): 
                        scipy_gen = stats.loguniform(para_min, para_max)
                        scipy_gen.random_state = self.MT 
                        self.gen_dict["Log"][f"{para_min}{para_max}" ] = scipy_gen
                    #
                    para_value = self.gen_dict["Log"][f"{para_min}{para_max}" ].rvs()
                # 
            #
            self.para_dict[idx] = para_value
        #
        return self.para_dict
    # 
    
    def sample_again(self):
        self.param_id = "{:07d}".format(rd.randint(0,10000000))
        self.hard_const = False 
        while not self.hard_const:
            self.sample_parameters()
            self.to_dimensionless()
            self.calculate_ss()
            self.hard_const = self.hard_constraints()
        # 
    
    def to_dimensionless(self): 
        K_TF    = self.para_dict["K_TF"]
        k_TFd   = self.para_dict["k_TFd"]
        k_dlt   = self.para_dict["k_dlt"]
        a0  = self.para_dict["a0"]
        a1  = self.para_dict["a1"]
        b0  = self.para_dict["b0"]
        rc  = self.para_dict["rc"]
        K   = self.para_dict["K"]
        k_d = self.para_dict["k_d"]
        V   = self.para_dict["V"]
        
        k_a = k_d / K 
        
        coef_M   = (rc + k_dlt)*V/a0
        coef_A   = (rc + k_dlt)**2*V/a0/a1
        coef_CTF = V 
        coef_t   = (rc + k_dlt)
        D_TF     = (k_TFd + k_dlt)/(rc + k_dlt)
        D        = (k_d + rc + k_dlt)/(rc + k_dlt) 
        sigma    = 0.05
        R        = k_a * a0 * a1 / (2 * V * (rc + k_dlt)**3)
        B0       = (b0 + k_dlt) / (rc + k_dlt) 
        kappa    = D / (4*R) 
        P        = 2 * R * K * D_TF / (K_TF*D)
        L        = (rc + k_dlt)**2 / (a0 * a1)
        
        self.para_dict.update( {
            "coef_M"    : coef_M,
            "coef_A"    : coef_A,
            "coef_CTF"  : coef_CTF, 
            "coef_t"    : coef_t, 
            "D_TF"      : D_TF, 
            "D"         : D, 
            "sigma"     : sigma, 
            "R"         : R, 
            "B0"        : B0, 
            "kappa"     : kappa, 
            "P"         : P, 
            "L"         : L
        } ) 
    # 

    def hard_constraints(self):
        D = self.para_dict["D"]
        sigma = self.para_dict["sigma"]
        freeAoverV = self.para_dict["L"] / (self.A_ss_bar - 2*self.A_2_ss_bar)
        A2overV = self.para_dict["L"] / (self.A_2_ss_bar)

        if freeAoverV > 0.1 : 
            print("Free A concentration is too small")
            return False 
        #
        if A2overV > 0.1 : 
            print("A2 dimer amount is too small")
            return False 
        #
        
        if sigma > 0.1 : 
            print("sigma over 0.1")
            return False 
        # 
        
        if D < 1 : 
            print("D less than 1")
            return False 
        # 
        
        if self.M_ss_bar > 10 * self.para_dict["coef_M"] / self.para_dict["V"]: 
            print("M steady state concentration is too large") 
            return False 
        # 
        
        if self.A_ss_bar > 10000 * self.para_dict["coef_A"] / self.para_dict["V"]:
            print("A steady state concentration too large")
            return False 
        #
        
        return True 
    #
    
    def calculate_ss(self):
        B0 = self.para_dict["B0"]
        nu = self.para_dict["nu"]
        sigma = self.para_dict["sigma"]
        kappa = self.para_dict["kappa"]
        
        self.M_ss_bar = 1/B0 * (nu+sigma) / (nu+1)
        self.A_ss_bar = self.M_ss_bar 
        self.A_2_ss_bar = 1/2 * ( self.A_ss_bar + kappa - np.sqrt(kappa * (2*self.A_ss_bar + kappa)) )
    #
#
