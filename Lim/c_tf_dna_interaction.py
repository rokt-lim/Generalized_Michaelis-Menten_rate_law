# -*- coding: utf-8 -*-
"""
C. TF-DNA interaction

A_bar = A_bar_max*(1.0 - 0.5*alpha_A_bar*(1.0 + cos(2.0*pi/k_delta/T*tau)))
A_bar = A_bar_max - 0.5*alpha_A_bar*A_bar_max - 0.5*alpha_A_bar*A_bar_max*cos(2.0*pi/k_delta/T*tau)

d(C_bar)/d(tau) = A_bar/(K*V) - (1.0 + A_bar)*C_bar

tau = k_delta*t
K = k_delta/k_a

dtheta = k_dlt/k_delta

A_bar = A/K
B_bar = (1/V)/K
C_bar = C/K
C_bar_QSSA = A_bar/(K*V*(1.0 + A_bar))

@author: LIM Roktaek
@e-mail: rokt.lim@gmail.com
"""

####-- Modules

import pandas
import numpy
from scipy.integrate import solve_ivp
from scipy.integrate import trapz

####-- END Modules

####-- Functions

###
def int_avg(t_eval,f_eval,n_period,T_period):
    t_area = trapz(f_eval,t_eval)
    v_avg = t_area/n_period/T_period
    #-- return
    return v_avg
###

###
def func_toy_X(X_max,alpha_X,T_coeff_X,phi_X,t_eval):
    #-- X = X_max*(1.0 - 0.5*alpha_X*(1.0 + numpy.cosine(T_coeff_X*t - phi_X)))
    #-- X = X_max - 0.5*alpha_X*X_max - 0.5*alpha_X*X_max*cos(T_coeff_X*t - phi_X)
    X_eval = X_max - 0.5*alpha_X*X_max - 0.5*alpha_X*X_max*numpy.cos(T_coeff_X*t_eval - phi_X)
    #-- return
    return X_eval
###

###
def func_toy_d_X(X_max,alpha_X,T_coeff_X,phi_X,t_eval):
    #-- d_X = 0.5*alpha_X*X_max*T_coeff_X*sin(T_coeff_X*t - phi_X)
    d_X_eval = 0.5*alpha_X*X_max*T_coeff_X*numpy.sin(T_coeff_X*t_eval - phi_X)
    #-- return
    return d_X_eval
###

###
def func_toy_dd_X(X_max,alpha_X,T_coeff_X,phi_X,t_eval):
    #-- dd_X = 0.5*alpha_X*X_max*T_coeff_X*T_coeff_X*cos(T_coeff_X*t - phi_X)
    dd_X_eval = 0.5*alpha_X*X_max*T_coeff_X*T_coeff_X*numpy.cos(T_coeff_X*t_eval - phi_X)
    #-- return
    return dd_X_eval
###

###
def func_mu_X(X_bar,d_X_bar):
    X_bar_inv = 1.0/X_bar
    mu_X = X_bar_inv*d_X_bar
    #-- return
    return mu_X
###

###
def func_d_mu_X(X_bar,d_X_bar,dd_X_bar):
    X_bar_inv = 1.0/X_bar
    d_mu_X = -X_bar_inv*X_bar_inv*d_X_bar*d_X_bar + X_bar_inv*dd_X_bar
    #-- return
    return d_mu_X
###

###
def func_d_C_tTF(C_tTF,tau_delay,mu_A,dtheta):
    #-- d_C_tTF = C_tTF/(1.0 + A_bar)*mu_A - dtheta*C_tTF
    #-- d_C_tTF = C_tTF*tau_delay*mu_A - dtheta*C_tTF
    d_C_tTF = C_tTF*tau_delay*mu_A - dtheta*C_tTF
    #-- return
    return d_C_tTF
###

###
def func_dd_C_tTF(C_tTF,tau_delay,A_bar,mu_A,d_mu_A):
    #-- dd_C_tTF = C_tTF/(1.0 + A_bar)*( (1.0 - A_bar)/(1.0 + A_bar)*mu_A*mu_A + d_mu_A)
    #-- dd_C_tTF = C_tTF*tau_delay*( (1.0 - A_bar)*tau_delay*mu_A*mu_A + d_mu_A)
    dd_C_tTF = C_tTF*tau_delay*((1.0 - A_bar)*tau_delay*mu_A*mu_A + d_mu_A)
    #-- return
    return dd_C_tTF
###

###
def A_bar_eval_n_func(df_coeff,t_eval):
    n_func = float(len(df_coeff.index.values))
    A_bar_eval = 0.0
    for idx in df_coeff.index.values:
        A_bar_eval = A_bar_eval + func_toy_X(df_coeff.loc[idx,'A_bar_max'],df_coeff.loc[idx,'alpha_A'],
                                             df_coeff.loc[idx,'T_coeff'],df_coeff.loc[idx,'phi_A'],t_eval)
    #
    A_bar_eval = A_bar_eval/float(n_func)
    #-- return
    return A_bar_eval
###

###
def func_C_TF_gamma(A_bar_max,alpha_A_bar,KV_inv,T_coeff,phi_A,tau_list,tau_delay):
    #-- C_TF_gamma
    df_TF_gamma = pandas.DataFrame({})
    df_TF_gamma['tau'] = tau_list
    df_TF_gamma['tau_delay'] = tau_delay
    df_TF_gamma['C_TF_gamma'] = 0.0
    df_TF_gamma['tau_gamma'] = 0.0
    for idx in df_TF_gamma.index.values:
        tau_d = df_TF_gamma.loc[idx,'tau'] - df_TF_gamma.loc[idx,'tau_delay']
        if tau_d >= 0.0:
            A_bar_tau_d = func_toy_X(A_bar_max,alpha_A_bar,T_coeff,phi_A,tau_d)
        else:
            A_bar_tau_d = func_toy_X(A_bar_max,alpha_A_bar,T_coeff,phi_A,0.0)
        #
        #-- C_TF_tQSSA = A_bar*KV_inv*(1.0 + A_bar)
        tmp = A_bar_tau_d*KV_inv/(1.0 + A_bar_tau_d)
        df_TF_gamma.loc[idx,'tau_gamma'] = tau_d
        df_TF_gamma.loc[idx,'C_TF_gamma'] = tmp
    #
    #-- return
    return df_TF_gamma['tau_gamma'].values,df_TF_gamma['C_TF_gamma'].values
###

###
def func_C_TF_gamma_irregular(tau_list,tau_delay,KV_inv,df_coeff):
    #-- C_TF_gamma, irregular
    df_v1 = pandas.DataFrame({})
    df_v1['tau'] = tau_list
    df_v1['tau_delay'] = tau_delay
    df_v1['C_TF_gamma'] = 0.0
    df_v1['tau_gamma'] = 0.0
    for idx in df_v1.index.values:
        tau_d = df_v1.loc[idx,'tau'] - df_v1.loc[idx,'tau_delay']
        A_bar_tau_d = A_bar_eval_n_func(df_coeff,tau_d)
        #-- C_TF_QSSA = A_bar*KV_inv*(1.0 + A_bar)
        tmp = A_bar_tau_d*KV_inv/(1.0 + A_bar_tau_d)
        df_v1.loc[idx,'tau_gamma'] = tau_d
        A_bar_tau = A_bar_eval_n_func(df_coeff,df_v1.loc[idx,'tau'])
        df_v1.loc[idx,'C_TF_gamma'] = numpy.amin([tmp,A_bar_tau,KV_inv])
    #
    #-- return
    return df_v1['tau_gamma'].values,df_v1['C_TF_gamma'].values
###

###
def ODE_TF_DNA_toy(t,y,pv):
    #-- d(C_bar)/d(tau) = A_bar/(K*V) - (1.0 + A_bar)*C_bar
    #-- KV_inv = 1.0/(K*V);
    #-- d(C_bar)/d(tau) = A_bar*KV_inv - (1.0 + A_bar)*C_bar;
    #-- A_bar = A_bar_max - 0.5*alpha_A_bar*A_bar_max - 0.5*alpha_A_bar*A_bar_max*cos(T_coeff*tau)
    #-- parameters
    # pv[0] = KV_inv = 1/(K_M*V)
    # pv[1] = A_bar_max
    # pv[2] = alpha_A
    # pv[3] = T_coeff = 2.0*numpy.pi/k_delta/T
    # pv[4] = phi_A
    #-- variables
    # A_bar = given
    # C_bar = y[0]
    A_bar = func_toy_X(pv[1],pv[2],pv[3],pv[4],t)
    #-- ordinary differential equation, Cbar
    dy_dt = pv[0]*A_bar - (1.0 + A_bar)*y
    #-- return
    return dy_dt
###

###
def ODE_TF_DNA_toy_irregular(t,y,pv):
    #-- d(C_bar)/d(tau) = A_bar/(K*V) - (1.0 + A_bar)*C_bar
    #-- KV_inv = 1.0/(K*V);
    #-- d(C_bar)/d(tau) = A_bar*KV_inv - (1.0 + A_bar)*C_bar;
    #-- A_bar = A_bar_max - 0.5*alpha_A_bar*A_bar_max - 0.5*alpha_A_bar*A_bar_max*cos(T_coeff*tau)
    #-- 10 A_bar functions
    #-- parameters
    #-- variables
    # A_bar = given, sum of 10 cosines
    # C_bar = y[0]
    A_bar_eval = A_bar_eval_n_func(pv,t)
    KV_inv =  pv.loc[0,'KV_inv']
    #-- ordinary differential equation, Cbar
    dy_dt = KV_inv*A_bar_eval - (1.0 + A_bar_eval)*y
    #-- return
    return dy_dt
###

###
def func_epsilon_TF(A_bar,tau_delay,mu_A):
    #-- epsilon_TF = A_bar/( (1.0 + A_bar)*(1.0 + A_bar) )*mu_A
    #-- epsilon_TF = A_bar*tau_delay*tau_delay*mu_A
    epsilon_TF = A_bar*tau_delay*tau_delay*mu_A
    epsilon_TF = numpy.absolute(epsilon_TF)
    #-- return
    return epsilon_TF
###

###
def func_epsilon_TF_Q(C_tTF,d_C_tTF,dd_C_tTF,tau_delay):
    #-- epsilon_TF_Q = (-1.0*d_C_tTF*tau_delay + dd_C_tTF*tau_delay*tau_delay)/C_tTF
    epsilon_TF_Q = (-1.0*d_C_tTF*tau_delay + dd_C_tTF*tau_delay*tau_delay)/C_tTF
    epsilon_TF_Q = numpy.absolute(epsilon_TF_Q)
    #-- return
    return epsilon_TF_Q
###

###
def solve_TF_DNA_interaction(t_start,t_end,dt,T_period,A_max,alpha_A,K_M,V,k_delta,phi_A):
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period: period of A(t), unit = h
    #-- A_max: maximum concentration of a transcription factor protein, unit = nM
    #-- alpha_A: the peak-to-trough difference of A(t)
    #-- K_M: the Michaelis constant, unit = nM
    #-- V: nuclear volume, unit = nM^-1
    #-- k_delta: unit = h^-1
    #
    #-- output
    #-- df: This dataframe has the values of A_bar, B_bar, C_bar, C_bar_QSSA, C_bar_gamma,
    #       epsilon_TF, epsilon_TF_gamma, and epsilon_TF_Q at the time points.
    #
    #-- parameters of the ODE for TF-DNA interaction
    # pv[0] = KV_inv = 1/(K_M*V)
    # pv[1] = A_bar_max
    # pv[2] = alpha_A
    # pv[3] = T_coeff = 2.0*numpy.pi/k_delta/T
    # pv[4] = phi_A = 0.0
    pv = [1.0/(K_M*V),A_max/K_M,alpha_A,2.0*numpy.pi/k_delta/T_period,phi_A]
    tp = k_delta*numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_TF_DNA_toy(t,y,pv),\
        t_span=[k_delta*t_start,k_delta*t_end],y0=[0.0],method='LSODA',\
        t_eval=tp,max_step=k_delta*dt)
    #-- save solutions
    df = pandas.DataFrame({})
    df['t'] = sol.t/k_delta
    df['tau'] = sol.t
    df['A_bar'] = func_toy_X(A_max/K_M,alpha_A,2.0*numpy.pi/k_delta/T_period,phi_A,sol.t)
    df['B_bar'] = 1.0/(K_M*V)
    df['C_bar'] = sol.y[0]
    df['KV'] = K_M*V
    df['KV_inv'] = 1.0/(K_M*V)
    df['k_delta'] = k_delta
    df['tau_delay'] = 1.0/(1.0 + df['A_bar'].values)
    df['C_bar_QSSA'] = df['A_bar'].values*df['KV_inv'].values*df['tau_delay'].values
    tau_gamma,C_TF_gamma = func_C_TF_gamma(A_max/K_M,alpha_A,1.0/(K_M*V),2.0*numpy.pi/k_delta/T_period,0.0,\
        df['tau'].values,df['tau_delay'].values)
    df['tau_gamma'] = tau_gamma
    df['C_bar_gamma'] = C_TF_gamma
    df['d_A_bar'] = func_toy_d_X(A_max/K_M,alpha_A,2.0*numpy.pi/k_delta/T_period,phi_A,sol.t)
    df['d_B_bar'] = 0.0
    df['dd_A_bar'] = func_toy_dd_X(A_max/K_M,alpha_A,2.0*numpy.pi/k_delta/T_period,phi_A,sol.t)
    df['dd_B_bar'] = 0.0
    df['mu_A'] = func_mu_X(df['A_bar'].values,df['d_A_bar'].values)
    df['mu_B'] = 0.0
    df['d_mu_A'] = func_d_mu_X(df['A_bar'].values,df['d_A_bar'].values,df['dd_A_bar'].values)
    df['d_mu_B'] = 0.0
    df['dtheta'] = 0.0
    df['d_C_tTF'] = func_d_C_tTF(df['C_bar_QSSA'].values,df['tau_delay'].values,df['mu_A'].values,df['dtheta'].values)
    df['dd_C_tTF'] = func_dd_C_tTF(df['C_bar_QSSA'].values,df['tau_delay'].values,df['A_bar'].values,df['mu_A'].values,df['d_mu_A'].values)
    #-- epsilon values
    df['epsilon_TF'] = func_epsilon_TF(df['A_bar'].values,df['tau_delay'].values,df['mu_A'].values)
    tmp_eval = numpy.absolute(0.5*df['tau_delay'].values*df['tau_delay'].values*df['dd_C_tTF'].values)
    df['epsilon_TF_gamma'] = tmp_eval/df['C_bar_gamma'].values
    df['epsilon_TF_Q'] = func_epsilon_TF_Q(df['C_bar_QSSA'].values,df['d_C_tTF'].values,df['dd_C_tTF'].values,df['tau_delay'].values)
    #-- save columns
    save_cols = ['tau','A_bar','B_bar','C_bar','C_bar_QSSA','C_bar_gamma','tau_gamma',\
        'epsilon_TF','epsilon_TF_gamma','epsilon_TF_Q']
    df = df.loc[:,save_cols]
    #-- return
    return df
###

###
def solve_TF_DNA_interaction_irregular(t_start,t_end,dt,A_max,alpha_A,K_M,V,k_delta,phi_A,T_A):
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- A_max: list of the maximum concentrations of a transcription factor protein, unit = nM
    #-- alpha_A: list of the peak-to-trough differences of A(t)
    #-- K_M: the Michaelis constant, unit = nM
    #-- V: nuclear volume, unit = nM^-1
    #-- k_delta: unit = h^-1
    #-- phi_A: list of the phases of A(t)
    #-- T_A: list of the periods of A(t), unit = h
    #
    #-- output
    #-- df: This dataframe has the values of A_bar, B_bar, C_bar, C_bar_QSSA, and C_bar_gamma at the time points.
    #
    #-- solve IVP
    pv = pandas.DataFrame({})
    pv['A_max'] = A_max
    pv['A_bar_max'] = pv['A_max'].values/K_M
    pv['V'] = V
    pv['alpha_A'] = alpha_A
    pv['K'] = K_M
    pv['KV'] = K_M*V
    pv['phi_A'] = phi_A
    pv['T_A'] = T_A
    pv['T_coeff'] = 2.0*numpy.pi/k_delta/pv['T_A'].values
    pv['KV_inv'] = 1.0/(K_M*V)
    tp = k_delta*numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_TF_DNA_toy_irregular(t,y,pv),\
        t_span=[k_delta*t_start,k_delta*t_end],y0=[0.0],method='LSODA',\
        t_eval=tp,max_step=k_delta*dt)
    #-- save solutions
    df = pandas.DataFrame({})
    df['t'] = sol.t/k_delta
    df['tau'] = sol.t
    df['A_bar'] = A_bar_eval_n_func(pv,sol.t)
    df['B_bar'] = 1.0/(K_M*V)
    df['C_bar'] = sol.y[0]
    df['K'] = K_M
    df['k_delta'] = k_delta
    df['KV_inv'] = 1.0/(K_M*V)
    df['tau_delay'] = 1.0/(1.0 + df['A_bar'].values)
    df['C_bar_QSSA'] = df['A_bar']*df['KV_inv']*df['tau_delay']
    tau_gamma,C_TF_gamma = func_C_TF_gamma_irregular(sol.t,df['tau_delay'].values,1.0/(K_M*V),pv)
    df['tau_gamma'] = tau_gamma
    df['C_bar_gamma'] = C_TF_gamma
    #-- save columns
    save_cols = ['tau','A_bar','B_bar','C_bar','C_bar_QSSA','C_bar_gamma','tau_gamma']
    df = df.loc[:,save_cols]
    #-- return
    return df
###

###
def main_C_TF_DNA_interaction():
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period: period of A(t), unit = h
    #-- A_max: maximum concentration of a transcription factor protein, unit = nM
    #-- alpha_A: the peak-to-trough difference of A(t)
    #-- K_M: the Michaelis constant, unit = nM
    #-- V: nuclear volume, unit = nM^-1
    #-- k_delta: unit = h^-1
    t_start = 0.0
    t_end = 480.0
    dt = 0.05
    T_period = 24.0
    A_max = 3.866309617
    alpha_A = 0.670295861
    K_M = 8.248556897
    V = 48.4116595
    k_delta = 0.278490094
    phi_A = 0.0
    df_sol = solve_TF_DNA_interaction(t_start,t_end,dt,T_period,A_max,alpha_A,K_M,V,k_delta,phi_A)
    df_sol.to_csv('test_TF_DNA_interaction.csv',index=False)
    #-- return
    return 0
###

###
def main_C_TF_DNA_interaction_irregular():
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- A_max: list of the maximum concentrations of a transcription factor protein, unit = nM
    #-- alpha_A: list of the peak-to-trough differences of A(t)
    #-- K_M: the Michaelis constant, unit = nM
    #-- V: nuclear volume, unit = nM^-1
    #-- k_delta: unit = h^-1
    #-- phi_A: list of the phases of A(t)
    #-- T_A: list of the periods of A(t), unit = h
    t_start = 0.0
    t_end = 480.0
    dt = 0.05
    A_max = [3.123910964,6.705389326,7.048925296,1.111976433,6.098815036,9.877968655,3.770623766,9.890969486,1.41109999,7.225034603]
    alpha_A = [0.786956868,0.857859872,0.741958425,0.725248026,0.666100109,0.884072634,0.618884649,0.704439001,0.664246484,0.79771686]
    K_M = 6.634187954
    V = 82.42712038
    k_delta = 0.187974698
    phi_A = [2.990623685,0.597991493,1.733233153,2.447228271,-0.494138186,1.254102807,0.835723883,0.321036524,1.188972034,-2.187287966]
    T_A = [30.96294879,29.75650619,38.49151011,36.36400602,29.71741075,26.55253858,17.59014592,21.57506722,23.88817557,25.79085197]
    df_sol = solve_TF_DNA_interaction_irregular(t_start,t_end,dt,A_max,alpha_A,K_M,V,k_delta,phi_A,T_A)
    df_sol.to_csv('test_TF_DNA_interaction_irregular.csv',index=False)
    #-- return
    return 0
###

####-- END Functions

####-- Main script

main_C_TF_DNA_interaction()
main_C_TF_DNA_interaction_irregular()

####-- END ---####