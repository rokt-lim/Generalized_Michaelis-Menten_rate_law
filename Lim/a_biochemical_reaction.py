# -*- coding: utf-8 -*-
"""
A. Biochemical reaction

d(A_bar)/d(tau) = -theta*C_bar
d(C_bar)/d(tau) = (A_bar - C_bar)*(B_bar - C_bar) - C_bar

In the code, delta_tQ is defined as follows:
delta_tQ = (1 + (A + B)/K)^2 - 4*A*B/(K^2).

@author: LIM Roktaek
@e-mail: rokt.lim@gmail.com
"""

####-- Modules

import copy
import pandas
import numpy
from scipy.integrate import solve_ivp

####-- END Modules

####-- Functions

###
def func_mu_A(theta,A_bar,C_bar):
    #-- mu_A = -r_c/k_delta*C_bar/A_bar = -theta*C_bar/A_bar
    mu_A = -theta*C_bar/A_bar
    #-- return
    return mu_A
###

###
def func_d_mu_A(theta,A_bar,C_bar,d_C_bar):
    #-- d(mu_A)/d(tau)
    d_mu_A = -theta*d_C_bar/A_bar - (theta*C_bar/A_bar)*(theta*C_bar/A_bar)
    #-- return
    return d_mu_A
###

###
def func_delta_tQ(A_bar,B_bar):
    #-- delta_tQ = (1.0 + A_bar + B_bar)*(1.0 + A_bar + B_bar) - 4.0*A_bar*B_bar
    delta_tQ_eval = (1.0 + A_bar + B_bar)*(1.0 + A_bar + B_bar) - 4.0*A_bar*B_bar
    #-- return
    return delta_tQ_eval
###

###
def func_d_C_bar(A_bar,B_bar,C_bar):
    #-- d(C_bar)/d(tau) = A_bar*B_bar - (1.0 + A_bar + B_bar)*C_bar + C_bar*C_bar
    d_C_bar = A_bar*B_bar - (1.0 + A_bar + B_bar)*C_bar + C_bar*C_bar
    #-- return
    return d_C_bar
###

###
def func_C_sQSSA(A_bar,B_bar):
    #-- sQSSA
    C_sQSSA = A_bar*B_bar/(1.0 + A_bar)
    #-- return
    return C_sQSSA
###

###
def func_C_tQSSA(A_bar,B_bar,sqrt_delta_tQ):
    #-- C_tQSSA = 0.5*(1.0 + A_bar + B_bar - sqrt_delta_tQ)
    C_tQSSA_eval = 0.5*(1.0 + A_bar + B_bar - sqrt_delta_tQ)
    #-- return
    return C_tQSSA_eval
###

###
def func_d_C_tQSSA(A_bar,B_bar,C_tQSSA,mu_A,mu_B,sqrt_delta_tQ_inv):
    tmp = (B_bar - C_tQSSA)*A_bar*mu_A + (A_bar - C_tQSSA)*B_bar*mu_B
    d_C_tQSSA_eval = sqrt_delta_tQ_inv*tmp
    #-- return
    return d_C_tQSSA_eval
###

###
def func_dd_C_tQSSA(A_bar,B_bar,C_tQSSA,mu_A,mu_B,d_mu_A,d_mu_B,delta_tQ,delta_tQ_inv,sqrt_delta_tQ_inv):
    tmp_1 = (delta_tQ - 2.0*A_bar*(1.0 + A_bar - C_tQSSA))*(B_bar - C_tQSSA)*A_bar*mu_A*mu_A
    tmp_2 = (delta_tQ - 2.0*B_bar*(1.0 + B_bar - C_tQSSA))*(A_bar - C_tQSSA)*B_bar*mu_B*mu_B
    tmp_3 = (1.0 + delta_tQ - (A_bar - B_bar)*(A_bar - B_bar))*A_bar*B_bar*mu_A*mu_B
    tmp_4 = delta_tQ*((B_bar - C_tQSSA)*A_bar*d_mu_A + (A_bar - C_tQSSA)*B_bar*d_mu_B)
    dd_C_tQSSA_eval = delta_tQ_inv*sqrt_delta_tQ_inv*(tmp_1 + tmp_2 + tmp_3 + tmp_4)
    #-- return
    return dd_C_tQSSA_eval
###

###
def func_C_bar_gamma(theta,tau_eval,tau_0,A_bar,B_bar,C_bar,C_tQSSA_0,C_tQSSA,sqrt_delta_tQ_inv):
    #-- copy the dataframe
    df_gamma = pandas.DataFrame({})
    df_gamma['tau'] = tau_eval
    df_gamma['A_bar'] = A_bar
    df_gamma['B_bar'] = B_bar
    df_gamma['C_bar'] = C_bar
    df_gamma['C_tQSSA'] = C_tQSSA
    df_gamma['sqrt_delta_tQ_inv'] = sqrt_delta_tQ_inv
    df_gamma['C_bar_gamma'] = 0.0
    df_gamma['tau_gamma'] = 0.0
    for idx in df_gamma.index.values:
        if df_gamma.loc[idx,'tau'] - df_gamma.loc[idx,'sqrt_delta_tQ_inv'] > tau_0:
            tau_d = df_gamma.loc[idx,'tau'] - df_gamma.loc[idx,'sqrt_delta_tQ_inv']
            idx_d = numpy.searchsorted(df_gamma['tau'].values,tau_d,side='right')
            #-- compute solution
            y_init = [df_gamma.loc[df_gamma.index.values[idx_d - 1],'A_bar'],df_gamma.loc[df_gamma.index.values[idx_d - 1],'C_bar']]
            sol = solve_ivp(fun=lambda t,y : ODE_enzyme_substrate_reaction(t,y,[df_gamma.loc[idx,'B_bar'],theta]),
                            t_span=[df_gamma.loc[df_gamma.index.values[idx_d - 1],'tau'],df_gamma.loc[df_gamma.index.values[idx_d],'tau']],
                            y0=y_init,method='LSODA',
                            t_eval=[df_gamma.loc[df_gamma.index.values[idx_d - 1],'tau'],tau_d,df_gamma.loc[df_gamma.index.values[idx_d],'tau']])
            sqrt_delta_tQ_tau_d = numpy.sqrt(func_delta_tQ(sol.y[0][1],df_gamma.loc[idx,'B_bar']))
            tmp = func_C_tQSSA(sol.y[0][1],df_gamma.loc[idx,'B_bar'],sqrt_delta_tQ_tau_d)
            #
            df_gamma.loc[idx,'tau_gamma'] = tau_d
        else:
            tmp = copy.deepcopy(C_tQSSA_0)
            df_gamma.loc[idx,'tau_gamma'] = tau_0
        #
        df_gamma.loc[idx,'C_bar_gamma'] = numpy.amin([tmp,df_gamma.loc[idx,'A_bar'],df_gamma.loc[idx,'B_bar']])
    #
    #-- return
    return df_gamma['C_bar_gamma'].values,df_gamma['tau_gamma'].values
###

###
def ODE_enzyme_substrate_reaction(t,y,pv):
    #-- ODEs
    # d(A_bar)/d(tau) = -theta*C_bar
    # d(C_bar)/d(tau) = (A_bar - C_bar)*(B_bar - C_bar) - C_bar
    # theta = r_c/k_delta, r_c = k_cat
    #-- parameters
    # pv[0] = B_bar
    # pv[1] = theta
    #-- variables
    # Abar = y[0]
    # Cbar = y[1]
    dy_dt = numpy.zeros(y.shape)
    #-- ordinary differential equations
    dy_dt[0] = -pv[1]*y[1]
    dy_dt[1] = (y[0] - y[1])*(pv[0] - y[1]) - y[1]
    #-- return
    return dy_dt
###

###
def func_epsilon_1(delta_tQ_inv,d_C_tQSSA):
    epsilon_1 = delta_tQ_inv*numpy.absolute(d_C_tQSSA)
    #-- return
    return epsilon_1
###

###
def func_epsilon_2(delta_tQ_inv,sqrt_delta_tQ_inv,A_bar,B_bar,mu_A,mu_B):
    epsilon_2 = delta_tQ_inv*sqrt_delta_tQ_inv*((1.0 + A_bar - B_bar)*A_bar*mu_A + (1.0 + B_bar - A_bar)*B_bar*mu_B)
    #-- return
    return numpy.absolute(epsilon_2)
###

###
def func_epsilon_tQ(delta_tQ_inv,sqrt_delta_tQ_inv,C_tQSSA,d_C_tQSSA,dd_C_tQSSA):
    epsilon_tQ = numpy.absolute(sqrt_delta_tQ_inv*d_C_tQSSA - delta_tQ_inv*dd_C_tQSSA)/C_tQSSA
    #-- return
    return epsilon_tQ
###

###
def func_epsilon_gamma(delta_tQ_inv,C_bar_gamma,dd_C_tQSSA):
    epsilon_gamma = 0.5*delta_tQ_inv/C_bar_gamma*numpy.absolute(dd_C_tQSSA)
    #-- return
    return epsilon_gamma
###

###
def enzyme_substrate_reaction_simulation(A_0,B_0,C_0,K_M,k_d,r_c,
                                         t_start,t_end,dt):
    #-- input
    #-- A_0: substrate concentration at t = 0, unit = mM
    #-- B_0: enzyme concentration at t = 0, unit = mM
    #-- C_0: enzyme-substrate complex concentration at t = 0
    #-- K_M: the Michaelis constant -> K_val
    #-- k_d: dissociation rate -> k_b
    #-- r_c: chemical conversion rate -> k_cat
    #
    #-- output
    #-- df_sol: This dataframe has the values of A_bar, B_bar, C_bar, C_bar_sQ, C_bar_tQ,
    #--         C_bar_gamma, epsilon_1, epsilon_2, epsilon_tQ, and epsilon_gamma at the time points.
    #
    #-- other constants
    #-- k_delta = k_d + r_c
    #-- theta = r_c/k_delta
    k_delta = k_d + r_c
    theta = r_c/k_delta
    A_bar_0 = A_0/K_M
    B_bar_0 = B_0/K_M
    C_bar_0 = C_0/K_M
    # tp = numpy.linspace(tau_start,tau_end,int((tau_end-tau_start)/dtau) + 1)
    tp = k_delta*numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1)
    pv = [B_bar_0,theta]
    #-- solve model
    sol = solve_ivp(fun=lambda t,y : ODE_enzyme_substrate_reaction(t,y,pv),\
        t_span=[t_start*k_delta,t_end*k_delta],y0=[A_bar_0,0.0],method='LSODA',\
        t_eval=tp,max_step=dt*k_delta)
    #-- data file
    df_sol = pandas.DataFrame({})
    df_sol['tau'] = sol.t
    df_sol['A_bar'] = sol.y[0]
    df_sol['B_bar'] = B_bar_0
    df_sol['C_bar'] = sol.y[1]
    df_sol['t'] = sol.t/k_delta
    df_sol['A'] = sol.y[0]*K_M
    df_sol['B'] = B_bar_0*K_M
    df_sol['C'] = sol.y[1]*K_M
    df_sol['d_C_bar'] = func_d_C_bar(df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['C_bar'].values)
    df_sol['delta_tQ'] = func_delta_tQ(df_sol['A_bar'].values,df_sol['B_bar'].values)
    df_sol['delta_tQ_inv'] = 1.0/df_sol['delta_tQ'].values
    df_sol['sqrt_delta_tQ'] = numpy.sqrt(df_sol['delta_tQ'].values)
    df_sol['sqrt_delta_tQ_inv'] = 1.0/df_sol['sqrt_delta_tQ'].values
    df_sol['C_bar_sQSSA'] = func_C_sQSSA(df_sol['A_bar'].values,df_sol['B_bar'].values)
    df_sol['C_sQSSA'] = df_sol['C_bar_sQSSA'].values*K_M
    df_sol['C_bar_tQSSA'] = func_C_tQSSA(df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['sqrt_delta_tQ'].values)
    df_sol['C_tQSSA'] = df_sol['C_bar_tQSSA'].values*K_M
    df_sol['mu_A'] = func_mu_A(theta,df_sol['A_bar'].values,df_sol['C_bar'].values)
    df_sol['mu_B'] = 0.0
    df_sol['d_mu_A'] = func_d_mu_A(theta,df_sol['A_bar'].values,df_sol['C_bar'].values,df_sol['d_C_bar'].values)
    df_sol['d_mu_B'] = 0.0
    df_sol['d_CtQSSA'] = func_d_C_tQSSA(df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['C_bar_tQSSA'].values,\
        df_sol['mu_A'].values,df_sol['mu_B'].values,df_sol['sqrt_delta_tQ_inv'].values)
    df_sol['dd_CtQSSA'] = func_dd_C_tQSSA(df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['C_bar_tQSSA'].values,\
        df_sol['mu_A'].values,df_sol['mu_B'].values,df_sol['d_mu_A'].values,df_sol['d_mu_B'].values,df_sol['delta_tQ'].values,\
        df_sol['delta_tQ_inv'].values,df_sol['sqrt_delta_tQ_inv'].values)
    C_bar_gamma,tau_gamma = func_C_bar_gamma(theta,df_sol['tau'].values,C_bar_0,\
        df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['C_bar'].values,
        df_sol.loc[0,'C_bar_tQSSA'],df_sol['C_bar_tQSSA'].values,df_sol['sqrt_delta_tQ_inv'].values)
    df_sol['C_bar_gamma'] = C_bar_gamma
    df_sol['C_gamma'] = C_bar_gamma*K_M
    df_sol['tau_gamma'] = tau_gamma
    df_sol['t_gamma'] = tau_gamma/k_delta
    #-- epsilon_1
    df_sol['epsilon_1'] = func_epsilon_1(df_sol['delta_tQ_inv'].values,df_sol['d_CtQSSA'].values)
    #-- epsilon_2
    df_sol['epsilon_2'] = func_epsilon_2(df_sol['delta_tQ_inv'].values,df_sol['sqrt_delta_tQ_inv'].values,\
        df_sol['A_bar'].values,df_sol['B_bar'].values,df_sol['mu_A'].values,df_sol['mu_B'].values)
    #-- pjaatQSSAcond
    df_sol['epsilon_tQ'] = func_epsilon_tQ(df_sol['delta_tQ_inv'].values,df_sol['sqrt_delta_tQ_inv'].values,\
        df_sol['C_bar_tQSSA'].values,df_sol['d_CtQSSA'].values,df_sol['dd_CtQSSA'].values)
    #-- epsilon_gamma
    df_sol['epsilon_gamma'] = func_epsilon_gamma(df_sol['delta_tQ_inv'].values,\
        df_sol['C_bar_gamma'].values,df_sol['dd_CtQSSA'].values)
    #-- columns
    save_cols = ['t','A','B','C','C_sQSSA','C_tQSSA','C_gamma','t_gamma',\
        'tau','A_bar','B_bar','C_bar','C_bar_sQSSA','C_bar_tQSSA','C_bar_gamma','tau_gamma',\
        'epsilon_1','epsilon_2','epsilon_tQ','epsilon_gamma']
    df_sol = df_sol.loc[:,save_cols]
    #-- return
    return df_sol
###

###
def main_A_biochemical_reaction():
    #-- concentration of oxaloacetate = 0.000486964 mM
    #-- concentration of malate dehydrogenase = 0.091 mM
    #-- concentration of complex = 0.0 mM
    #-- the Michaelis constant, K_M = 0.04 mM
    #-- dissociation rate, k_d = 0.0 1/sec
    #-- chemical conversion rate, r_c = 931 1/sec
    A_0 = 0.000486964
    B_0 = 0.091
    C_0 = 0.0
    K_M = 0.04
    k_d = 0.0
    r_c = 931.0
    t_start = 0.0
    t_end = 0.005
    dt = 0.001/1000.0
    out_file = 'malate_dehydrogenase-oxaloacetate-reaction.csv'
    df_sol = enzyme_substrate_reaction_simulation(A_0,B_0,C_0,K_M,k_d,r_c,t_start,t_end,dt)
    df_sol.to_csv(out_file,index=False)
    #-- return
    return 0
###

####-- END Functions

####-- Main script

main_A_biochemical_reaction()

####-- END ---####