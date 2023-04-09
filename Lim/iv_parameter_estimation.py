# -*- coding: utf-8 -*-
"""
IV. Parameter estimation

In the code, delta_tQ is defined as follows:
delta_tQ = (K + A + B)^2 - 4*A*B.

@author: LIM Roktaek
@e-mail: rokt.lim@gmail.com
"""

####-- Modules

import pandas
import numpy
from scipy.integrate import solve_ivp
from scipy import optimize

####-- END Modules

####-- Functions

###
import sys
def error_exit():
    checker = input('Enter "x" to finish this program: ')
    if checker == 'x':
        sys.exit()
    #
    #-- return
    return 0
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
def ODE_PPI_toy_tau(t,y,pv):
    #-- d(C_bar)/d(tau) = A_bar*B_bar - A_bar*C_bar - B_bar*C_bar + C_bar*C_bar - C_bar
    #-- A_bar = A_bar_max - 0.5*alpha_A_bar*A_bar_max - 0.5*alpha_A_bar*A_bar_max*cos(T_coeff_A_bar*tau - phi_A)
    #-- B_bar = B_bar_max - 0.5*alpha_B_bar*B_bar_max - 0.5*alpha_B_bar*B_bar_max*cos(T_coeff_B_bar*tau - phi_B)
    #-- parameters
    # pv[0] = A_bar_max
    # pv[1] = alpha_A_bar
    # pv[2] = T_coeff_A_bar
    # pv[3] = phi_A
    A_bar = func_toy_X(pv[0],pv[1],pv[2],pv[3],t)
    # pv[4] = B_bar_max
    # pv[5] = alpha_B_bar
    # pv[6] = T_coeff_B_bar
    # pv[7] = phi_B
    B_bar = func_toy_X(pv[4],pv[5],pv[6],pv[7],t)
    #-- variables
    # A_bar = given
    # B_bar = given
    # C_bar = y[0]
    #-- ordinary differential equation, C_bar
    dy_dt = A_bar*B_bar - A_bar*y - B_bar*y + y*y - y
    #-- return
    return dy_dt
###

###
def solve_ODE_PPI_toy_tau(t_start,t_end,dt,k_delta,C_bar_init,pv,tau_eval):
    # pv = [A_bar_max,alpha_A_bar,T_coeff_A,phi_A,
    #       B_bar_max,alpha_B_bar,T_coeff_B,phi_B]
    sol = solve_ivp(fun=lambda t,y : ODE_PPI_toy_tau(t,y,pv),\
                    t_span=[k_delta*t_start,k_delta*t_end],y0=[C_bar_init],method='LSODA',\
                    t_eval=tau_eval,max_step=k_delta*dt)
    #-- return
    return sol
###

###
def func_C_PPI_tQSSA(A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B,K,t_list):
    #-- delta_tQ = (K + A + B)*(K + A + B) - 4.0*A*B
    #-- C_tQSSA = 0.5*(K + A + B - sqrt(delta_tQ))
    A_eval = func_toy_X(A_max,alpha_A,T_coeff_A,phi_A,t_list)
    B_eval = func_toy_X(B_max,alpha_B,T_coeff_B,phi_B,t_list)
    delta_tQ = (K + A_eval + B_eval)*(K + A_eval + B_eval) - 4.0*A_eval*B_eval
    C_PPI_tQSSA = 0.5*(K + A_eval + B_eval - numpy.sqrt(delta_tQ))
    #-- return
    return C_PPI_tQSSA
###

###
def func_C_PPI_tQSSA_der_K(A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B,K,t_list):
    #-- delta_tQ = (K + A + B)*(K + A + B) - 4.0*A*B
    #-- C_tQSSA = 0.5*(K + A + B - sqrt((K + A + B)^2 - 4*A*B))
    #-- d(C_tQSSA)/d(K) = 0.5 - 0.5*(K + A + B)/sqrt(delta_tQ)
    A_eval = func_toy_X(A_max,alpha_A,T_coeff_A,phi_A,t_list)
    B_eval = func_toy_X(B_max,alpha_B,T_coeff_B,phi_B,t_list)
    delta_tQ = (K + A_eval + B_eval)*(K + A_eval + B_eval) - 4.0*A_eval*B_eval
    delta_tQ_sqrt_inv = 1.0/numpy.sqrt(delta_tQ)
    C_PPI_tQSSA_der_K = 0.5 - 0.5*delta_tQ_sqrt_inv*(K + A_eval + B_eval)
    #-- return
    return C_PPI_tQSSA_der_K
###

###
def func_C_bar_PPI_gamma(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,
                         B_bar_max,alpha_B_bar,T_coeff_B_bar,phi_B_bar,
                         tau_list,tau_delay):
    #-- gamma
    df_gamma = pandas.DataFrame({})
    df_gamma['tau'] = tau_list
    df_gamma['tau_delay'] = tau_delay
    df_gamma['C_bar_PPI_gamma'] = 0.0
    for idx in df_gamma.index.values:
        #-- A_bar and B_bar at tau
        A_bar_tau = func_toy_X(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,df_gamma.loc[idx,'tau'])
        B_bar_tau = func_toy_X(B_bar_max,alpha_B_bar,T_coeff_B_bar,phi_B_bar,df_gamma.loc[idx,'tau'])
        #-- A_bar and B_bar at tau_d
        tau_d = df_gamma.loc[idx,'tau'] - df_gamma.loc[idx,'tau_delay']
        A_bar_tau_d = func_toy_X(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,tau_d)
        B_bar_tau_d = func_toy_X(B_bar_max,alpha_B_bar,T_coeff_B_bar,phi_B_bar,tau_d)
        delta_tQ = (1.0 + A_bar_tau_d + B_bar_tau_d)*(1.0 + A_bar_tau_d + B_bar_tau_d) - 4.0*A_bar_tau_d*B_bar_tau_d
        #-- C_bar_PPI_gamma(tau - tau_delay)
        tmp = 0.5*(1.0 + A_bar_tau_d + B_bar_tau_d - numpy.sqrt(delta_tQ))
        df_gamma.loc[idx,'C_bar_PPI_gamma'] = numpy.amin([tmp,A_bar_tau,B_bar_tau])
    #
    #-- return
    return df_gamma['C_bar_PPI_gamma'].values
###

###
def fmin_C_PPI_tQSSA_lsq(param,t_data,C_data,pv_PPI_tQSSA):
    #-- param = K
    #-- pv_PPI_tQSSA = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    C_PPI_tQSSA = func_C_PPI_tQSSA(pv_PPI_tQSSA[0],pv_PPI_tQSSA[1],pv_PPI_tQSSA[2],pv_PPI_tQSSA[3],
                                   pv_PPI_tQSSA[4],pv_PPI_tQSSA[5],pv_PPI_tQSSA[6],pv_PPI_tQSSA[7],
                                   param,t_data)
    #-- return sum (y_data - y_est)*(y_data - y_est)
    sum_sq = numpy.sum((C_data - C_PPI_tQSSA)*(C_data - C_PPI_tQSSA))
    #-- return
    return sum_sq
###

###
def parameter_estimation_PPI_tQSSA_lsq_powell(param,bnds,t_data,C_data,pv_tQSSA):
    #-- param = K
    #-- pv_PPI_tQSSA = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    res_tQSSA = optimize.minimize(fmin_C_PPI_tQSSA_lsq,[param],method='Powell',bounds=bnds,
                                  args=(t_data,C_data,pv_tQSSA))
    tQSSA_est = ['{:.8e}'.format(res_tQSSA.fun),str(res_tQSSA.nfev),str(res_tQSSA.nit),str(res_tQSSA.success)]
    #-- return
    return res_tQSSA.x,tQSSA_est
###

###
def residual_C_PPI_tQSSA_trf(param,t_data,C_data,pv_PPI_tQSSA):
    #-- param = K
    #-- This returns the vector of residuals: r_i = C_i - C_tQSSA(t_i,K).
    #-- pv_PPI_tQSSA = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    C_PPI_tQSSA = func_C_PPI_tQSSA(pv_PPI_tQSSA[0],pv_PPI_tQSSA[1],pv_PPI_tQSSA[2],pv_PPI_tQSSA[3],
                                   pv_PPI_tQSSA[4],pv_PPI_tQSSA[5],pv_PPI_tQSSA[6],pv_PPI_tQSSA[7],
                                   param,t_data)
    #-- return, C_data - C_PPI_tQSSA
    return C_data - C_PPI_tQSSA
###

###
def der_K_residual_C_PPI_tQSSA_trf(param,t_data,C_data,pv_PPI_tQSSA):
    #-- param = K
    #-- This computes the partial derivatives of r_i with respect to K, d(r_i)/d(K) = -d(C_tQSSA(t_i,K))/d(K)
    #-- pv_PPI_tQSSA = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    C_PPI_tQSSA_der_K = func_C_PPI_tQSSA_der_K(pv_PPI_tQSSA[0],pv_PPI_tQSSA[1],pv_PPI_tQSSA[2],pv_PPI_tQSSA[3],
                                               pv_PPI_tQSSA[4],pv_PPI_tQSSA[5],pv_PPI_tQSSA[6],pv_PPI_tQSSA[7],
                                               param,t_data)
    der_K_residual = -1.0*C_PPI_tQSSA_der_K
    #-- return
    return numpy.array(numpy.hsplit(der_K_residual,len(C_data)))
###

###
def parameter_estimation_PPI_tQSSA_lsq_trf(param,bnds,t_data,C_data,pv_tQSSA):
    #-- param = K
    #-- pv_PPI_tQSSA = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    res_tQSSA = optimize.least_squares(residual_C_PPI_tQSSA_trf,param,jac=der_K_residual_C_PPI_tQSSA_trf,\
        bounds=bnds,args=(t_data,C_data,pv_tQSSA))
    tQSSA_est = ['{:.8e}'.format(res_tQSSA.optimality),str(res_tQSSA.nfev),str(res_tQSSA.njev),str(res_tQSSA.success)]
    #-- return
    return res_tQSSA.x,tQSSA_est
###

###
def fmin_C_PPI_gamma_lsq(params,t_data,C_data,pv_PPI_gamma):
    #-- params[0] = K
    #-- params[1] = k_delta
    #-- pv_PPI_gamma = [A_max,alpha_A,T_coeff_A,phi_A,B_max,alpha_B,T_coeff_B,phi_B]
    #-- C_PPI_gamma = C_PPI_tQSSA(tau - tau_delay)
    #-- tau_delay = 1.0/sqrt(delta_tQ)
    K_inv = 1.0/params[0]
    A_bar_eval = func_toy_X(pv_PPI_gamma[0]*K_inv,pv_PPI_gamma[1],pv_PPI_gamma[2]/params[1],pv_PPI_gamma[3],t_data*params[1])
    B_bar_eval = func_toy_X(pv_PPI_gamma[4]*K_inv,pv_PPI_gamma[5],pv_PPI_gamma[6]/params[1],pv_PPI_gamma[7],t_data*params[1])
    delta_tQ = (1.0 + A_bar_eval + B_bar_eval)*(1.0 + A_bar_eval + B_bar_eval) - 4.0*A_bar_eval*B_bar_eval
    tau_delay = 1.0/numpy.sqrt(delta_tQ)
    C_bar_gamma = func_C_bar_PPI_gamma(pv_PPI_gamma[0]*K_inv,pv_PPI_gamma[1],pv_PPI_gamma[2]/params[1],pv_PPI_gamma[3],
                                       pv_PPI_gamma[4]*K_inv,pv_PPI_gamma[5],pv_PPI_gamma[6]/params[1],pv_PPI_gamma[7],
                                       t_data*params[1],tau_delay)
    #-- return sum (y_data - y_est)*(y_data - y_est)
    sum_sq = numpy.sum((C_data - C_bar_gamma*params[0])*(C_data - C_bar_gamma*params[0]))
    #-- return
    return sum_sq
###

###
def parameter_estimation_PPI_gamma_lsq_powell(params,bnds,t_data,C_data,pv_PPI_gamma):
    #-- params[0] = K
    #-- params[1] = k_delta
    res_gamma = optimize.minimize(fmin_C_PPI_gamma_lsq,params,method='Powell',bounds=bnds,
                                  args=(t_data,C_data,pv_PPI_gamma))
    gamma_est = ['{:.8e}'.format(res_gamma.fun),str(res_gamma.nfev),str(res_gamma.nit),str(res_gamma.success)]
    #-- return
    return res_gamma.x[0],res_gamma.x[1],gamma_est
###

###
def parameter_estimation_ppi(t_start,t_end,dt,T_period,
                             A_bar_max,alpha_A_bar,phi_A,B_bar_max,alpha_B_bar,phi_B,K_M,k_delta,
                             df_p_init):
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of A(t) and B(t)
    #-- A_bar_max: maximum dimensionless concentration of A
    #-- alpha_A_bar: the peak-to-trough difference of A_bar
    #-- B_bar_max: maximum dimensionless concentration of A
    #-- alpha_B_bar: the peak-to-trough difference of B_bar
    #-- phi_B: the phase difference between A_bar and B_bar
    #-- K_M: the Michaelis constant, unit = nM
    #-- k_delta: unit = h^-1
    #-- time constants
    T_coeff_t = 2.0*numpy.pi/T_period
    C_bar_init = 0.0
    #-- data generation for parameter estimation, last 3 periods, time step size = 2 hour
    pv = [A_bar_max,alpha_A_bar,T_coeff_t/k_delta,phi_A,
          B_bar_max,alpha_B_bar,T_coeff_t/k_delta,phi_B]
    t_eval = numpy.linspace(408.0,480.0,num=37)
    sol = solve_ODE_PPI_toy_tau(t_start,t_end,dt,k_delta,C_bar_init,pv,t_eval*k_delta)
    t_data = sol.t/k_delta
    C_data = sol.y[0]*K_M
    pv_PPI_tQSSA = [A_bar_max*K_M,alpha_A_bar,T_coeff_t,phi_A,
                    B_bar_max*K_M,alpha_B_bar,T_coeff_t,phi_B]
    bnds_K=[(0.01,1000.0)]
    bnds_K_trf = (0.01,1000.0)
    bnds_K_k_delta = [(0.01,1000.0),(0.1,10.0)]
    df_p_init['K_tQSSA_powell'] = 0.0
    df_p_init['K_tQSSA_powell_res'] = 0.0
    df_p_init['tQSSA_powell_status'] = False
    df_p_init['K_tQSSA_trf'] = 0.0
    df_p_init['K_tQSSA_trf_res'] = 0.0
    df_p_init['tQSSA_trf_status'] = False
    df_p_init['K_gamma_powell'] = 0.0
    df_p_init['k_delta_gamma_powell'] = 0.0
    df_p_init['gamma_powell_res'] = 0.0
    df_p_init['gamma_powell_status'] = False
    for p_idx in df_p_init.index.values:
        #-- tQSSA, objective function = least squares, optimization method = Powell
        K_tQSSA,est_tQSSA = parameter_estimation_PPI_tQSSA_lsq_powell(df_p_init.loc[p_idx,'K_init'],\
            bnds_K,t_data,C_data,pv_PPI_tQSSA)
        df_p_init.loc[p_idx,'K_tQSSA_powell'] = K_tQSSA
        df_p_init.loc[p_idx,'K_tQSSA_powell_res'] = float(est_tQSSA[0])
        df_p_init.loc[p_idx,'tQSSA_powell_status'] = est_tQSSA[3]
        #-- tQSSA, objective function = least squares, optimization method = Trust Region Reflective algorithm
        K_tQSSA,est_tQSSA = parameter_estimation_PPI_tQSSA_lsq_trf(df_p_init.loc[p_idx,'K_init'],\
            bnds_K_trf,t_data,C_data,pv_PPI_tQSSA)
        df_p_init.loc[p_idx,'K_tQSSA_trf'] = K_tQSSA
        df_p_init.loc[p_idx,'K_tQSSA_trf_res'] = float(est_tQSSA[0])
        df_p_init.loc[p_idx,'tQSSA_trf_status'] = est_tQSSA[3]
        #-- gamma, objective function = least squares, optimization method = Powell
        tmp_p_init = [df_p_init.loc[p_idx,'K_init'],df_p_init.loc[p_idx,'k_delta_init']]
        K_gamma,k_delta_gamma,est_gamma = parameter_estimation_PPI_gamma_lsq_powell(tmp_p_init,\
            bnds_K_k_delta,t_data,C_data,pv_PPI_tQSSA)
        df_p_init.loc[p_idx,'K_gamma_powell'] = K_gamma
        df_p_init.loc[p_idx,'k_delta_gamma_powell'] = k_delta_gamma
        df_p_init.loc[p_idx,'gamma_powell_res'] = float(est_gamma[0])
        df_p_init.loc[p_idx,'gamma_powell_status'] = est_gamma[3]
    #
    #-- best estimations of K, tQSSA, Powell
    idx_tmp = df_p_init[ df_p_init['tQSSA_powell_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        K_tQSSA_powell_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_powell_res'].idxmin()
        K_tQSSA_powell = df_p_init.loc[idx_min,'K_tQSSA_powell']
    else:
        K_tQSSA_powell_status = False
        K_tQSSA_powell = 0.0
    #
    #-- best estimations of K, tQSSA, Trust Region Reflective algorithm
    idx_tmp = df_p_init[ df_p_init['tQSSA_trf_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        K_tQSSA_trf_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_trf_res'].idxmin()
        K_tQSSA_trf = df_p_init.loc[idx_min,'K_tQSSA_trf']
    else:
        K_tQSSA_trf_status = False
        K_tQSSA_trf = 0.0
    #
    #-- best estimations of K, gamma, Powell
    idx_tmp = df_p_init[ df_p_init['gamma_powell_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        gamma_powell_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_powell_res'].idxmin()
        K_gamma_powell = df_p_init.loc[idx_min,'K_gamma_powell']
        k_delta_gamma_powell = df_p_init.loc[idx_min,'k_delta_gamma_powell']
    else:
        gamma_powell_status = False
        K_gamma_powell = 0.0
        k_delta_gamma_powell = 0.0
    #
    #-- output
    output = { 'K_tQSSA_powell_status' : K_tQSSA_powell_status,
               'K_tQSSA_powell'        : K_tQSSA_powell,
               'K_tQSSA_trf_status'    : K_tQSSA_trf_status,
               'K_tQSSA_trf'           : K_tQSSA_trf,
               'gamma_powell_status'   : gamma_powell_status,
               'K_gamma_powell'        : K_gamma_powell,
               'k_delta_gamma_powell'  : k_delta_gamma_powell }
    #-- return
    return output
###

###
def ODE_TF_DNA_toy_tau(t,y,pv):
    #-- d(C_bar)/d(tau) = A_bar/(K*V) - C_bar - A_bar*C_bar
    #-- KV_inv = 1.0/(K*V);
    #-- d(C_bar)/d(tau) = KV_inv*A_bar - C_bar - A_bar*C_bar
    #-- A_bar = A_bar_max - 0.5*alpha_A_bar*A_bar_max - 0.5*alpha_A_bar*A_bar_max*cos(T_coeff_A_bar*tau - phi_A_bar)
    #-- parameters
    # pv[0] = KV_inv
    # pv[1] = A_bar_max
    # pv[2] = alpha_A_bar
    # pv[3] = T_coeff_A_bar
    # pv[4] = phi_A_bar
    A_bar = func_toy_X(pv[1],pv[2],pv[3],pv[4],t)
    #-- variables
    # A_bar = given
    # C_bar = y[0]
    #-- ordinary differential equation, C_bar
    dy_dt = pv[0]*A_bar - y - A_bar*y
    #-- return
    return dy_dt
###

###
def solve_ODE_TF_DNA_toy_tau(t_start,t_end,dt,k_delta,C_bar_init,pv):
    # pv = [KV_inv,A_bar_max,alpha_A_bar,T_coeff_A,phi_A_bar]
    tau_eval = k_delta*numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_TF_DNA_toy_tau(t,y,pv),\
                    t_span=[k_delta*t_start,k_delta*t_end],y0=[C_bar_init],method='LSODA',\
                    t_eval=tau_eval,max_step=k_delta*dt)
    #-- return
    return sol
###

###
def func_C_TF_QSSA(A_max,alpha_A,T_coeff_A,phi_A,K,V_inv,t_list):
    #-- C_bar_TF_QSSA = A_bar/(K*V*(1.0 + A_bar))
    #-- C_TF_QSSA = A/(K*V*(1.0 + A/K)) = A/(V*(K + A))
    A_eval = func_toy_X(A_max,alpha_A,T_coeff_A,phi_A,t_list)
    C_TF_QSSA_dem = 1.0/(K + A_eval)*V_inv
    C_TF_QSSA = A_eval*C_TF_QSSA_dem
    #-- return
    return C_TF_QSSA
###

###
def func_C_TF_QSSA_der_K(A_max,alpha_A,T_coeff_A,phi_A,K,V_inv,t_list):
    #-- C_TF_QSSA = A/(K*V*(1.0 + A/K)) = A/(V*(K + A))
    #-- d(C_TF_QSSA)/d(K) = -1.0*A/(V*(K + A)*(K + A))
    A_eval = func_toy_X(A_max,alpha_A,T_coeff_A,phi_A,t_list)
    C_TF_QSSA_dem_K = 1.0/(K + A_eval)
    C_TF_QSSA_der_K = -1.0*A_eval*C_TF_QSSA_dem_K*C_TF_QSSA_dem_K*V_inv
    #-- return
    return C_TF_QSSA_der_K
###

###
def func_C_bar_TF_gamma(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,KV_inv,tau_list,tau_delay):
    #-- gamma
    df_gamma = pandas.DataFrame({})
    df_gamma['tau'] = tau_list
    df_gamma['tau_delay'] = tau_delay
    df_gamma['C_bar_TF_gamma'] = 0.0
    for idx in df_gamma.index.values:
        tau_d = df_gamma.loc[idx,'tau'] - df_gamma.loc[idx,'tau_delay']
        A_bar_tau_d = func_toy_X(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,tau_d)
        #-- C_TF_QSSA = A_bar*KV_inv*(1.0 + A_bar)
        tmp = A_bar_tau_d*KV_inv/(1.0 + A_bar_tau_d)
        A_bar_tau = func_toy_X(A_bar_max,alpha_A_bar,T_coeff_A_bar,phi_A_bar,df_gamma.loc[idx,'tau'])
        df_gamma.loc[idx,'C_bar_TF_gamma'] = numpy.amin([tmp,A_bar_tau,KV_inv])
    #
    #-- return
    return df_gamma['C_bar_TF_gamma'].values
###

###
def fmin_C_TF_QSSA_lsq(param,t_data,C_data,pv_TF_QSSA):
    #-- param = K
    #-- C_tQSSA = A/(V*(K + A))
    #-- pv_TF_QSSA = [A_max,alpha_A,T_coeff_A,phi_A,V_inv]
    C_TF_QSSA = func_C_TF_QSSA(pv_TF_QSSA[0],pv_TF_QSSA[1],pv_TF_QSSA[2],pv_TF_QSSA[3],
                                 param,pv_TF_QSSA[4],t_data)
    #-- return sum (y_data - y_est)*(y_data - y_est)
    sum_sq = numpy.sum((C_data - C_TF_QSSA)*(C_data - C_TF_QSSA))
    #-- return
    return sum_sq
###

###
def parameter_estimation_TF_QSSA_lsq_powell(param,bnds,t_data,C_data,pv_tQSSA):
    #-- param = K
    #-- pv_TF_QSSA = [A_max,alpha_A,T_coeff_A,phi_A,1/V]
    res_tQSSA = optimize.minimize(fmin_C_TF_QSSA_lsq,[param],method='Powell',bounds=bnds,
                                  args=(t_data,C_data,pv_tQSSA))
    tQSSA_est = ['{:.8e}'.format(res_tQSSA.fun),str(res_tQSSA.nfev),str(res_tQSSA.nit),str(res_tQSSA.success)]
    #-- return
    return res_tQSSA.x,tQSSA_est
###

###
def residual_C_TF_QSSA_trf(param,t_data,C_data,pv_TF_QSSA):
    #-- param = K
    #-- This returns the vector of residuals: r_i = C_i - C_TF_QSSA(t_i,K).
    #-- pv_TF_QSSA = [A_max,alpha_A,T_coeff_A,phi_A,1/V]
    C_TF_QSSA = func_C_TF_QSSA(pv_TF_QSSA[0],pv_TF_QSSA[1],pv_TF_QSSA[2],pv_TF_QSSA[3],
                                 param,pv_TF_QSSA[4],t_data)
    #-- return, C_data - C_PPI_tQSSA
    return C_data - C_TF_QSSA
###

###
def der_K_residual_C_TF_QSSA_trf(param,t_data,C_data,pv_TF_QSSA):
    #-- param = K
    #-- This computes the partial derivatives of r_i with respect to K, d(r_i)/d(K) = -d(C_TF_QSSA(t_i,K))/d(K)
    #-- pv_TF_QSSA = [A_max,alpha_A,T_coeff_A,phi_A,1/V]
    C_TF_QSSA_der_K = func_C_TF_QSSA_der_K(pv_TF_QSSA[0],pv_TF_QSSA[1],pv_TF_QSSA[2],pv_TF_QSSA[3],
                                             param,pv_TF_QSSA[4],t_data)
    der_K_residual = -1.0*C_TF_QSSA_der_K
    #-- return
    return numpy.array(numpy.hsplit(der_K_residual,len(C_data)))
###

###
def parameter_estimation_TF_QSSA_lsq_trf(param,bnds,t_data,C_data,pv_tQSSA):
    #-- param = K
    #-- pv_TF_QSSA = [A_max,alpha_A,T_coeff_A,phi_A,1/V]
    res_tQSSA = optimize.least_squares(residual_C_TF_QSSA_trf,param,jac=der_K_residual_C_TF_QSSA_trf,\
        bounds=bnds,args=(t_data,C_data,pv_tQSSA))
    tQSSA_est = ['{:.8e}'.format(res_tQSSA.optimality),str(res_tQSSA.nfev),str(res_tQSSA.njev),str(res_tQSSA.success)]
    #-- return
    return res_tQSSA.x,tQSSA_est
###

###
def fmin_C_TF_gamma_lsq(params,t_data,C_data,pv_TF_gamma):
    #-- params[0] = K
    #-- params[1] = k_delta
    #-- pv_TF_gamma = [A_max,alpha_A,T_coeff_A,phi_A,V_inv]
    #-- TF_v1, C_TF_QSSA(tau - tau_delay)
    #-- tau_delay = 1.0/(1.0 + A_bar), A_bar = A/K
    K_inv = 1.0/params[0]
    A_eval = func_toy_X(pv_TF_gamma[0],pv_TF_gamma[1],pv_TF_gamma[2],pv_TF_gamma[3],t_data)
    tau_delay = 1.0/(1.0 + A_eval*K_inv)
    C_bar_TF_gamma = func_C_bar_TF_gamma(pv_TF_gamma[0]*K_inv,pv_TF_gamma[1],pv_TF_gamma[2]/params[1],pv_TF_gamma[3],\
        K_inv*pv_TF_gamma[4],params[1]*t_data,tau_delay)
    #-- return sum (y_data - y_est)*(y_data - y_est)
    sum_sq = numpy.sum((C_data - C_bar_TF_gamma*params[0])*(C_data - C_bar_TF_gamma*params[0]))
    return sum_sq
###

###
def parameter_estimation_TF_gamma_lsq_powell(params,bnds,t_data,C_data,pv_TF_gamma):
    #-- params[0] = K
    #-- params[1] = k_delta
    res_gamma = optimize.minimize(fmin_C_TF_gamma_lsq,params,method='Powell',bounds=bnds,
                                  args=(t_data,C_data,pv_TF_gamma))
    gamma_est = ['{:.8e}'.format(res_gamma.fun),str(res_gamma.nfev),str(res_gamma.nit),str(res_gamma.success)]
    #-- return
    return res_gamma.x[0],res_gamma.x[1],gamma_est
###

###
def parameter_estimation_tf(t_start,t_end,dt,T_period,
                            A_max,alpha_A,phi_A,V,K_M,k_delta,
                            df_p_init):
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of A(t)
    #-- A_max: maximum concentration of A, unit = nM
    #-- alpha_A: the peak-to-trough difference of A
    #-- V: nuclear volume, unit = nM^-1
    #-- K_M: the Michaelis constant, unit = nM
    #-- k_delta: unit = h^-1
    #-- time constants
    T_coeff_t = 2.0*numpy.pi/T_period
    C_bar_init = 0.0
    #-- data generation for parameter estimation, last 3 periods, time step size = 2 hour
    KV_inv = 1.0/(K_M*V)
    pv = [KV_inv,A_max/K_M,alpha_A,T_coeff_t/k_delta,phi_A]
    sol = solve_ODE_TF_DNA_toy_tau(t_start,t_end,dt,k_delta,C_bar_init,pv)
    t_data = sol.t/k_delta
    C_data = sol.y[0]*K_M
    pv_TF_QSSA = [A_max,alpha_A,T_coeff_t,phi_A,1.0/V]
    bnds_K=[(0.01,1000.0)]
    bnds_K_trf = (0.01,1000.0)
    bnds_K_k_delta = [(0.01,1000.0),(0.1,10.0)]
    df_p_init['K_tQSSA_powell'] = 0.0
    df_p_init['K_tQSSA_powell_res'] = 0.0
    df_p_init['tQSSA_powell_status'] = False
    df_p_init['K_tQSSA_trf'] = 0.0
    df_p_init['K_tQSSA_trf_res'] = 0.0
    df_p_init['tQSSA_trf_status'] = False
    df_p_init['K_gamma_powell'] = 0.0
    df_p_init['k_delta_gamma_powell'] = 0.0
    df_p_init['gamma_powell_res'] = 0.0
    df_p_init['gamma_powell_status'] = False
    for p_idx in df_p_init.index.values:
        #-- tQSSA, objective function = least squares, optimization method = Powell
        K_tQSSA,est_tQSSA = parameter_estimation_TF_QSSA_lsq_powell(df_p_init.loc[p_idx,'K_init'],\
            bnds_K,t_data,C_data,pv_TF_QSSA)
        df_p_init.loc[p_idx,'K_tQSSA_powell'] = K_tQSSA
        df_p_init.loc[p_idx,'K_tQSSA_powell_res'] = float(est_tQSSA[0])
        df_p_init.loc[p_idx,'tQSSA_powell_status'] = est_tQSSA[3]
        #-- tQSSA, objective function = least squares, optimization method = Trust Region Reflective algorithm
        K_tQSSA,est_tQSSA = parameter_estimation_TF_QSSA_lsq_trf(df_p_init.loc[p_idx,'K_init'],\
            bnds_K_trf,t_data,C_data,pv_TF_QSSA)
        df_p_init.loc[p_idx,'K_tQSSA_trf'] = K_tQSSA
        df_p_init.loc[p_idx,'K_tQSSA_trf_res'] = float(est_tQSSA[0])
        df_p_init.loc[p_idx,'tQSSA_trf_status'] = est_tQSSA[3]
        #-- gamma, objective function = least squares, optimization method = Powell
        tmp_p_init = [df_p_init.loc[p_idx,'K_init'],df_p_init.loc[p_idx,'k_delta_init']]
        K_gamma,k_delta_gamma,est_gamma = parameter_estimation_TF_gamma_lsq_powell(tmp_p_init,\
            bnds_K_k_delta,t_data,C_data,pv_TF_QSSA)
        df_p_init.loc[p_idx,'K_gamma_powell'] = K_gamma
        df_p_init.loc[p_idx,'k_delta_gamma_powell'] = k_delta_gamma
        df_p_init.loc[p_idx,'gamma_powell_res'] = float(est_gamma[0])
        df_p_init.loc[p_idx,'gamma_powell_status'] = est_gamma[3]
    #
    #-- best estimations of K, tQSSA, Powell
    idx_tmp = df_p_init[ df_p_init['tQSSA_powell_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        K_tQSSA_powell_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_powell_res'].idxmin()
        K_tQSSA_powell = df_p_init.loc[idx_min,'K_tQSSA_powell']
    else:
        K_tQSSA_powell_status = False
        K_tQSSA_powell = 0.0
    #
    #-- best estimations of K, tQSSA, Trust Region Reflective algorithm
    idx_tmp = df_p_init[ df_p_init['tQSSA_trf_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        K_tQSSA_trf_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_trf_res'].idxmin()
        K_tQSSA_trf = df_p_init.loc[idx_min,'K_tQSSA_trf']
    else:
        K_tQSSA_trf_status = False
        K_tQSSA_trf = 0.0
    #
    #-- best estimations of K, gamma, Powell
    idx_tmp = df_p_init[ df_p_init['gamma_powell_status'] == 'True' ].index.values
    if len(idx_tmp) != 0:
        gamma_powell_status = True
        idx_min = df_p_init.loc[idx_tmp,'K_tQSSA_powell_res'].idxmin()
        K_gamma_powell = df_p_init.loc[idx_min,'K_gamma_powell']
        k_delta_gamma_powell = df_p_init.loc[idx_min,'k_delta_gamma_powell']
    else:
        gamma_powell_status = False
        K_gamma_powell = 0.0
        k_delta_gamma_powell = 0.0
    #
    #-- output
    output = { 'K_tQSSA_powell_status' : K_tQSSA_powell_status,
               'K_tQSSA_powell'        : K_tQSSA_powell,
               'K_tQSSA_trf_status'    : K_tQSSA_trf_status,
               'K_tQSSA_trf'           : K_tQSSA_trf,
               'gamma_powell_status'   : gamma_powell_status,
               'K_gamma_powell'        : K_gamma_powell,
               'k_delta_gamma_powell'  : k_delta_gamma_powell }
    #-- return
    return output
###

###
def main_IV_p_estimation_ppi():
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of A(t) and B(t)
    #-- A_bar_max: maximum dimensionless concentration of A
    #-- alpha_A_bar: the peak-to-trough difference of A_bar
    #-- B_bar_max: maximum dimensionless concentration of A
    #-- alpha_B_bar: the peak-to-trough difference of B_bar
    #-- phi_B: the phase difference between A_bar and B_bar
    #-- K_M: the Michaelis constant, unit = nM
    #-- k_delta: unit = h^-1
    t_start = 0.0
    t_end = 480.0
    dt = 0.05
    T_period = 24.0
    A_bar_max = 32.79482405
    alpha_A_bar = 0.645539383
    phi_A = 0.0
    B_bar_max = 15.41806579
    alpha_B_bar = 0.494697832
    phi_B = 3.077723456
    K_M = 0.069567625
    k_delta = 0.226924353
    #-- parameter esitmation
    n_init = 10
    df_p_init = pandas.DataFrame({})
    df_p_init['K_init'] = numpy.power(10.0,numpy.random.uniform(-2.0,3.0,n_init))
    df_p_init['k_delta_init'] = numpy.power(10.0,numpy.random.uniform(-1.0,1.0,n_init))
    p_est_result = parameter_estimation_ppi(t_start,t_end,dt,T_period,\
        A_bar_max,alpha_A_bar,phi_A,B_bar_max,alpha_B_bar,phi_B,K_M,k_delta,df_p_init)
    for item in p_est_result.keys():
        print('\n' + item + ' : ' + str(p_est_result[item]))
    #
    #-- return
    return 0
###

###
def main_IV_p_estimation_tf():
    #-- input
    #-- Constants for ODE: t_start, t_end, dt
    #-- T_period : period of A(t)
    #-- A_max: maximum concentration of A, unit = nM
    #-- alpha_A: the peak-to-trough difference of A
    #-- V: nuclear volume, unit = nM^-1
    #-- K_M: the Michaelis constant, unit = nM
    #-- k_delta: unit = h^-1
    t_start = 0.0
    t_end = 480.0
    dt = 0.05
    T_period = 24.0
    A_max = 3.046706124
    alpha_A_bar = 0.212955484
    phi_A = 0.0
    V = 17.43002349
    K_M = 0.896697557
    k_delta = 9.500758797
    #-- parameter esitmation
    n_init = 10
    df_p_init = pandas.DataFrame({})
    df_p_init['K_init'] = numpy.power(10.0,numpy.random.uniform(-2.0,3.0,n_init))
    df_p_init['k_delta_init'] = numpy.power(10.0,numpy.random.uniform(-1.0,1.0,n_init))
    p_est_result = parameter_estimation_tf(t_start,t_end,dt,T_period,\
        A_max,alpha_A_bar,phi_A,V,K_M,k_delta,df_p_init)
    for item in p_est_result.keys():
        print('\n' + item + ' : ' + str(p_est_result[item]))
    #
    #-- return
    return 0
###

####-- END Functions

####-- Main script

main_IV_p_estimation_ppi()
# main_IV_p_estimation_tf()

####-- END ---####