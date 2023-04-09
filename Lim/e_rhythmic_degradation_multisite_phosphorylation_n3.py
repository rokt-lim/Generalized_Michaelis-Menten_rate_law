# -*- coding: utf-8 -*-
"""
E. Rhythmic degradation of proteins, number of post-translational modifications = 3

dA(t)/dt   = g(t) - r_c*A_3(t)
dA_1(t)/dt = k_0*B*(A(t) - A_1(t) - A_2(t) - A_3(t)) - k_1*B*A_1(t)
dA_2(t)/dt = k_1*B*A_1(t) - k_2*B*A_2(t)
dA_3(t)/dt = k_2*B*A_2(t) - r_c*A_3(t)

A(t)   : total concentration of protein
A_1(t) : concentration of the 1st modified protein
A_2(t) : concentration of the 2nd modified protein
A_3(t) : concentration of the 3rd modified protein
g(t)   : protein synthesis rate
r_c    : modified protein's turnover rate
B      : ubiquitin ligase or kinase concentration
k_0    : 1st modification rate coefficient
k_1    : 2nd modification rate coefficient
k_2    : 3rd modification rate coefficient

@author: LIM Roktaek
@e-mail: rokt.lim@gmail.com
"""

####-- Modules

import pandas
import numpy
from scipy.integrate import solve_ivp
from scipy.interpolate import splrep
from scipy.interpolate import splev

####-- END Modules

####-- Functions

###
def func_toy_X(X_max,alpha_X,T_coeff_X,phi_X,t_eval):
    #-- X = X_max*(1.0 - 0.5*alpha_X*(1.0 + numpy.cosine(T_coeff_X*t - phi_X)))
    #-- X = X_max - 0.5*alpha_X*X_max - 0.5*alpha_X*X_max*cos(T_coeff_X*t - phi_X)
    X_eval = X_max - 0.5*alpha_X*X_max - 0.5*alpha_X*X_max*numpy.cos(T_coeff_X*t_eval - phi_X)
    #-- return
    return X_eval
###

###
def multisite_phosphorylation_Q(K_eval,B_eval,A_eval):
    coeff = B_eval/(K_eval + B_eval)
    QSSA = coeff*A_eval
    #-- return
    return QSSA
###

###
def multisite_phosphorylation_gamma(A_sp,t_eval,gamma_delay,K_eval,B_eval):
    coeff = B_eval/(K_eval + B_eval)
    A_delay = splev(t_eval - gamma_delay,A_sp,der=0)
    gamma = coeff*A_delay
    #-- return
    return gamma
###

###
def ODE_rhythmic_degradation_g_t_n3(t,y,pv):
    dy_dt = numpy.zeros(y.shape)
    #-- dA/dt = g(t) - r_c*A_3(t)
    #-- g(t) = g_max - 0.5*alpha_g*g_max - 0.5*alpha_g*g_max*cos(T_coeff*t - phi_g), phi_g = 0.0
    #-- dA_1/dt = k_0*B*(A - A_1 - A_2 - A_3) - k_1*B*A_1
    #-- dA_2/dt = k_1*B*A_1 - k_2*B*A_2
    #-- dA_3/dt = k_2*B*A_2 - r_c*A_3
    #-- parameters
    # pv[0] = g_max
    # pv[1] = alpha_g
    # pv[2] = T_coeff_t = 2.0*numpy.pi/T_period
    # pv[3] = phi_g
    #-- variables
    # g(t) and B(t) are given
    g_eval = func_toy_X(pv[0],pv[1],pv[2],pv[3],t)
    # pv[4] = B_eval
    # pv[5] = r_c
    # pv[6] = k_2
    # pv[7] = k_1
    # pv[8] = k_0
    #-- variable
    # A   = y[0]
    # A_1 = y[1]
    # A_2 = y[2]
    # A_3 = y[3]
    #-- ordinary differential equations
    dy_dt = numpy.zeros(y.shape)
    dy_dt[0] = g_eval - pv[5]*y[3]
    dy_dt[1] = pv[8]*pv[4]*(y[0] - y[1] - y[2] - y[3]) - pv[7]*pv[4]*y[1]
    dy_dt[2] = pv[7]*pv[4]*y[1] - pv[6]*pv[4]*y[2]
    dy_dt[3] = pv[6]*pv[4]*y[2] - pv[5]*y[3]
    #-- return
    return dy_dt
###

###
def solve_rhythmic_degradation_g_t_n3(t_start,t_end,dt,T_period,g_max,alpha_g,B_eval,r_c,k_2,k_1,k_0):
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of g(t), unit = h
    #-- g_max    : maximum protein synthesis rate, unit = nM/h
    #-- alpha_g  : the peak-to-trough difference of g(t)
    #-- B_eval   : ubiquitin ligase or kinase concentration, unit = nM
    #-- r_c      : modified protein's turnover rate, unit = 1/h
    #-- k_0      : 1st modification rate coefficient, unit = nM^-1 h^-1
    #-- k_1      : 2nd modification rate coefficient, unit = nM^-1 h^-1
    #-- k_2      : 3rd modification rate coefficient, unit = nM^-1 h^-1
    #
    #-- parameters for the system of ODEs
    # pv[0] = g_max
    # pv[1] = alpha_g
    # pv[2] = 2.0*numpy.pi/T_period
    # pv[3] = 0.0
    # pv[4] = B_eval
    # pv[5] = r_c
    # pv[6] = k_2
    # pv[7] = k_1
    # pv[8] = k_0
    #-- initial conditions for A, A_1, A_2, and A_3
    A_init = 1.0
    A_1_init = 0.0
    A_2_init = 0.0
    A_3_init = 0.0
    #-- solve the system of ODEs
    pv = [g_max,alpha_g,2.0*numpy.pi/T_period,0.0,B_eval,r_c,k_2,k_1,k_0]
    t_eval = numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_rhythmic_degradation_g_t_n3(t,y,pv),\
        t_span=[t_start,t_end],y0=[A_init,A_1_init,A_2_init,A_3_init],method='LSODA',\
        t_eval=t_eval,max_step=dt)
    #-- save the solution
    df = pandas.DataFrame({})
    df['t'] = sol.t
    df['A'] = sol.y[0]
    df['A_1'] = sol.y[1]
    df['A_2'] = sol.y[2]
    df['A_3'] = sol.y[3]
    df['B'] = B_eval
    #-- dA/dt = g(t) - r_c*A_3(t)
    df['d_A'] = func_toy_X(g_max,alpha_g,2.0*numpy.pi/T_period,0.0,sol.t) - r_c*df['A_3'].values
    df['A_inv'] = 1.0/df['A'].values
    df['R_t'] = -1.0*df['d_A'].values*df['A_inv'].values
    df['r_t'] = r_c*df['A_3'].values*df['A_inv'].values
    #-- coefficients
    K_1_eval = k_1*B_eval/k_0
    K_2_eval = (k_2 - k_1)*B_eval/k_1
    K_3_eval = (r_c - k_2*B_eval)/k_2
    tmp = k_0*B_eval + k_1*B_eval
    gamma_1_delay = 1.0/tmp
    gamma_2_delay = 1.0/(k_2*B_eval)
    gamma_3_delay = 1.0/r_c
    #-- r_t, QSSA
    A_1_Q = multisite_phosphorylation_Q(K_1_eval,B_eval,df['A'].values)
    A_2_Q = multisite_phosphorylation_Q(K_2_eval,B_eval,A_1_Q)
    A_3_Q = multisite_phosphorylation_Q(K_3_eval,B_eval,A_2_Q)
    df['r_t_Q'] = r_c*A_3_Q*df['A_inv'].values
    #-- r_t, gamma
    A_sp = splrep(df['t'].values,df['A'].values,k=3)
    A_1_gamma = multisite_phosphorylation_gamma(A_sp,df['t'].values,gamma_1_delay,K_1_eval,B_eval)
    A_1_gamma_sp = splrep(df['t'].values,A_1_gamma,k=3)
    A_2_gamma = multisite_phosphorylation_gamma(A_1_gamma_sp,df['t'].values,gamma_2_delay,K_2_eval,B_eval)
    A_2_gamma_sp = splrep(df['t'].values,A_2_gamma,k=3)
    A_3_gamma = multisite_phosphorylation_gamma(A_2_gamma_sp,df['t'].values,gamma_3_delay,K_3_eval,B_eval)
    df['r_t_gamma'] = r_c*A_3_gamma*df['A_inv'].values
    #-- save columns
    save_cols = ['t','A','B','A_1','A_2','A_3','R_t','r_t','r_t_Q','r_t_gamma']
    df = df.loc[:,save_cols]
    #-- return
    return df
###

###
def main_E_rhythmic_degradation_n3():
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of g(t), unit = h
    #-- g_max    : maximum protein synthesis rate, unit = nM/h
    #-- alpha_g  : the peak-to-trough difference of g(t)
    #-- B_eval   : ubiquitin ligase or kinase concentration, unit = nM
    #-- r_c      : modified protein's turnover rate, unit = 1/h
    #-- k_0      : 1st modification rate coefficient, unit = nM^-1 h^-1
    #-- k_1      : 2nd modification rate coefficient, unit = nM^-1 h^-1
    #-- k_2      : 3rd modification rate coefficient, unit = nM^-1 h^-1
    t_start = 0.0
    t_end = 720.0
    dt = 0.05
    T_period = 24.0
    g_max = 1.835569285
    alpha_g = 1.0
    B_eval = 89.41785691
    r_c = 4.301739957
    k_2 = 0.032196664
    k_1 = 0.016535068
    k_0 = 0.056942427
    df_sol = solve_rhythmic_degradation_g_t_n3(t_start,t_end,dt,T_period,g_max,alpha_g,B_eval,r_c,k_2,k_1,k_0)
    df_sol.to_csv('test_rhythmic_degradation_n3.csv',index=False)
    #-- return
    return 0
###

####-- END Functions

####-- Main script

main_E_rhythmic_degradation_n3()

####-- END ---####