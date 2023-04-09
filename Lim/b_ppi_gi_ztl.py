# -*- coding: utf-8 -*-
"""
B. Protein-protein interaction, GI-ZTL interaction

k_delta = r_c + k_d
         dC(t)/dt = k_a*(A(t) - C(t))*(B(t) - C(t)) - k_delta*C(t)
                  = k_a*A(t)*B(t) - k_a*A(t)*C(t) - k_a*B(t)*C(t) + k_a*C(t)*C(t) - k_delta*C(t)
d(A(t) - C(t))/dt = g_A - k_a*(A(t) - C(t))*(B(t) - C(t)) + k_d*C(t) - r_A*(A(t) - C(t))
       d(A(t))/dt = d(C(t))/dt + g_A - k_a*(A(t) - C(t))*(B(t) - C(t)) + k_d*C(t) - r_A*(A(t) - C(t))
                  = g_A + k_d*C(t) - r_A*(A(t) - C(t)) - k_delta*C(t)
                  = g_A + (r_A - r_c)*C(t) - r_A*A(t)

In the code, delta_tQ is defined as follows:
delta_tQ = (K + A + B)^2 - 4*A*B.

@author: LIM Roktaek
@e-mail: rokt.lim@gmail.com
"""

####-- Modules

import copy
import pandas
import numpy
from scipy.integrate import trapz
from scipy.integrate import solve_ivp
from scipy.interpolate import splrep
from scipy.interpolate import splev

from ddeint import ddeint

####-- END Modules

####-- Functions

###
def protein_level_spline(data_file,x_tag,y_tag,y_fac,spline_order):
    p_data = pandas.read_csv(data_file)
    x_pt_rep = copy.deepcopy(p_data[x_tag].values[0:-1])
    y_pt_rep = copy.deepcopy(y_fac*p_data[y_tag].values[0:-1])
    #-- copies of time and level
    n_rep = 1000
    for i_rep in range(0,n_rep):
        x_pt_rep = numpy.concatenate((p_data[x_tag].values[0:-1] - 48.0*float(i_rep+1),x_pt_rep))
        y_pt_rep = numpy.concatenate((y_pt_rep,y_fac*p_data[y_tag].values[0:-1]))
    #
    for i_rep in range(0,n_rep):
        x_pt_rep = numpy.concatenate((x_pt_rep,p_data[x_tag].values[0:-1] + 48.0*float(i_rep+1)))
        y_pt_rep = numpy.concatenate((y_pt_rep,y_fac*p_data[y_tag].values[0:-1]))
    #
    f_sp = splrep(x_pt_rep,y_pt_rep,k=spline_order)
    #-- return
    return f_sp
###

###
def phase_shift(func_1,func_2,dt,f_period):
    func1_tmp = func_1 - numpy.mean(func_1)
    func2_tmp = func_2 - numpy.mean(func_2)
    tmp = dt*(numpy.argmax(numpy.correlate(func1_tmp,func2_tmp,mode='same')) \
        - numpy.argmax(numpy.correlate(func1_tmp,func1_tmp,mode='same')))
    if (tmp < 0.0) and (abs(tmp) > 0.5*f_period):
        p_shift = f_period + tmp
    elif (tmp > 0.0) and (abs(tmp) > 0.5*f_period):
        p_shift = tmp - f_period
    else:
        p_shift = tmp
    #
    #-- return
    return p_shift
###

###
def similarity(t_eval,f_v,approx_v):
    max_fun = numpy.maximum(f_v,approx_v)
    max_area = trapz(max_fun,t_eval)
    min_fun = numpy.minimum(f_v,approx_v)
    min_area = trapz(min_fun,t_eval)
    sim = min_area/max_area
    #-- return
    return sim
###

###
def ODE_GI_ZTL_full(t,y,pv):
    #-- dA/dt = g_A + (r_A - r_c)*C(t) - r_A*A(t)
    #-- dC/dt = k_a_L*A(t)*B(t) - k_a_L*A(t)*C(t) - k_a_L*B(t)*C(t) + k_a_L*C(t)*C(t) - k_delta_L*C(t) -- light
    #-- dC/dt = k_a_D*A(t)*B(t) - k_a_D*A(t)*C(t) - k_a_D*B(t)*C(t) + k_a_D*C(t)*C(t) - k_delta_D*C(t) -- darkness
    #-- k_a_D < k_a_L, k_a_L = k_a_D + k_a_d
    #-- dC/dt = (k_a_D + k_a_d)*A(t)*B(t) - (k_a_D + k_a_d)*A(t)*C(t) - (k_a_D + k_a_d)*B(t)*C(t) + (k_a_D + k_a_d)*C(t)*C(t)
    #--        - k_delta_L*C(t) -- light
    #-- dC/dt = k_a_D*A(t)*B(t) - k_a_D*A(t)*C(t) - k_a_D*B(t)*C(t) + k_a_D*C(t)*C(t)
    #--        - (k_delta_L + k_delta_d)*C(t) -- darkness
    #-- B(t) is given.
    #-- parameters
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = k_a_D
    # pv[5] = k_delta_L
    # pv[6] = k_a_d = k_a_L - k_a_D
    # pv[7] = k_delta_d = k_delta_D - k_delta_L
    #-- variables
    # y[0] = A(t)
    # y[1] = C(t)
    #-- ordinary differential equation, A(t) and C(t)
    B_sp = pv[0]
    B_eval = splev(t,B_sp,der=0)
    dy_dt = numpy.zeros(y.shape)
    #-- A(t)
    dy_dt[0] = pv[1] + pv[2]*y[1] - (pv[2] + pv[3])*y[0]
    #-- C(t)
    if t%24.0 < 12.0:
        dy_dt[1] = (pv[4] + pv[6])*(y[0] - y[1])*(B_eval - y[1]) - pv[5]*y[1]
    else:
        dy_dt[1] = pv[4]*(y[0] - y[1])*(B_eval - y[1]) - (pv[5] + pv[7])*y[1]
    #
    #-- return
    return dy_dt
###

###
def func_delta_tQ(K_eval,A_eval,B_eval):
    delta_tQ_eval = (K_eval + A_eval + B_eval)*(K_eval + A_eval + B_eval) - 4.0*A_eval*B_eval
    #-- return
    return numpy.absolute(delta_tQ_eval)
###

###
def func_C_tQSSA(K_eval,A_eval,B_eval,delta_tQ):
    C_tQSSA_eval = 0.5*(K_eval + A_eval + B_eval - numpy.sqrt(delta_tQ))
    #-- return
    return C_tQSSA_eval
###

###
def ODE_GI_ZTL_tQSSA(t,y,pv):
    #-- dA/dt = g_A + (r_A - r_c)*C(t) - (r_A - r_c + r_c)*A(t)
    #-- C(t) ~ C_tQSSA(t)
    #-- C_tQSSA(t) = 0.5*(K + A + B - sqrt(delta_tQ))
    #-- delta_tQ = (K + A + B)*(K + A + B) - 4.0*A*B
    #-- B(t) is given.
    #-- parameters
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = K_L
    # pv[5] = k_L_d
    #-- variables
    # y[0] = A(t)
    #-- ordinary differential equation, A(t) and C(t)
    B_sp = pv[0]
    B_eval = splev(t,B_sp,der=0)
    dy_dt = numpy.zeros(y.shape)
    #-- A(t)
    if t%24.0 < 12.0:
        delta_tQ = func_delta_tQ(pv[4],y[0],B_eval)
        C_tQSSA = func_C_tQSSA(pv[4],y[0],B_eval,delta_tQ)
        dy_dt[0] = pv[1] + pv[2]*C_tQSSA - (pv[2] + pv[3])*y[0]
    else:
        delta_tQ = func_delta_tQ(pv[4] + pv[5],y[0],B_eval)
        C_tQSSA = func_C_tQSSA(pv[4] + pv[5],y[0],B_eval,delta_tQ)
        dy_dt[0] = pv[1] + pv[2]*C_tQSSA - (pv[2] + pv[3])*y[0]
    #
    #-- return
    return dy_dt
###

###
def ODE_GI_ZTL_gamma(A_func,t,pv):
    #-- dA/dt = g_A + (r_A - r_c)*C(t) - (r_A - r_c + r_c)*A(t)
    #-- C(t) ~ C_gamma(t) = C_tQSSA(t - tau_delay/k_delta)
    #-- C_tQSSA(t) = 0.5*(K + A + B - sqrt(delta_tQ))
    #-- delta_tQ = (K + A + B)*(K + A + B) - 4.0*A*B
    #-- tau_delay = 1/sqrt(delta_tQ_tau)
    #-- delta_tQ_tau = (1.0 + A_bar + B_bar) - 4.0*A_bar*B_bar
    #-- A_bar = A/K, B_bar = B/K
    #-- B(t) is given.
    #-- parameters
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = K_L
    # pv[5] = k_L_d = K_D - K_L
    # pv[6] = k_delta_L
    # pv[7] = k_delta_d = k_delta_D - k_delta_L
    #-- variables
    # y[0] = A(t)
    #-- ordinary differential equation, A(t) and C(t)
    B_sp = pv[0]
    B_eval = splev(t,B_sp,der=0)
    A_eval = A_func(t)
    #-- A(t)
    if t%24.0 < 12.0:
        #-- delay
        delta_tQ = func_delta_tQ(pv[4],A_eval,B_eval)
        K_sq_inv = 1.0/(pv[4]*pv[4])
        delta_tQ_tau = delta_tQ*K_sq_inv
        t_delay = 1.0/numpy.sqrt(delta_tQ_tau)/pv[6]
        #-- C_gamma
        A_d = A_func(t - t_delay)
        if t - t_delay > 0.0:
            B_d = splev(t - t_delay,B_sp,der=0)
        else:
            B_d = splev(0.0,B_sp,der=0)
        #
        delta_tQ_d = func_delta_tQ(pv[4],A_d,B_d)
        C_tQSSA_d = func_C_tQSSA(pv[4],A_d,B_d,delta_tQ_d)
        C_gamma = numpy.amin([C_tQSSA_d,A_eval,B_eval])
        dy_dt = pv[1] + pv[2]*C_gamma - (pv[2] + pv[3])*A_eval
    else:
        #-- delay
        delta_tQ = func_delta_tQ(pv[4] + pv[5],A_eval,B_eval)
        K_sq_inv = 1.0/((pv[4] + pv[5])*(pv[4] + pv[5]))
        delta_tQ_tau = delta_tQ*K_sq_inv
        t_delay = 1.0/numpy.sqrt(delta_tQ_tau)/(pv[6] + pv[7])
        #-- C_gamma
        A_d = A_func(t - t_delay)
        if t - t_delay > 0.0:
            B_d = splev(t - t_delay,B_sp,der=0)
        else:
            B_d = splev(0.0,B_sp,der=0)
        #
        delta_tQ_d = func_delta_tQ(pv[4] + pv[5],A_d,B_d)
        C_tQSSA_d = func_C_tQSSA(pv[4] + pv[5],A_d,B_d,delta_tQ_d)
        C_gamma = numpy.amin([C_tQSSA_d,A_eval,B_eval])
        dy_dt = pv[1] + pv[2]*C_gamma - (pv[2] + pv[3])*A_eval
    #
    #-- return
    return dy_dt
###

###
def GI_ZTL_full(t_start,t_end,dt,ZTL_data_file,GI_data_file,
                w_ZTL,w_GI,g_A,r_A,r_c,k_d_L,k_d_D,K_L,K_D):
    #-- input
    #-- ODE solver parameters: t_start,t_end,dt
    #-- normalized protein level of ZTL: normalized_ZTL_data.csv
    #-- normalized protein level of GI : normalized_GI_data.csv
    #-- w_ZTL : scaling coefficient of ZTL, unit = nM
    #-- w_GI  : scaling coefficient of GI, unit = nM
    #-- g_A   : ZTL synthesis rate
    #-- r_A   : degradation rate of free ZTL
    #-- r_c   : degradation rate of GI-binding ZTL
    #-- k_d_L : dissociation rate of GI-binding ZTL in light
    #-- k_d_D : dissociation rate of GI-binding ZTL in darkness
    #-- K_L   : the Michaelis constant in light
    #-- K_D   : the Michaelis constant in darkness
    #
    #-- output
    #-- normalized protein levels of GI at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated by solving the system of ODEs
    #
    #-- Read data file, normalized protein level of ZTL
    ZTL_sp = protein_level_spline(ZTL_data_file,'time','ZTL_norm',1.0,3)
    ZTL_data = pandas.read_csv(ZTL_data_file)
    ZTL_y_data = ZTL_data['ZTL_norm'].values
    #-- Read data file, normalized protein level of GI
    GI_sp = protein_level_spline(GI_data_file,'time','GI_norm',w_GI,3)
    #-- k_delta = r_c + k_d
    #-- k_delta_L = r_c + k_d_L, k_delta in light
    #-- k_delta_D = r_c + k_d_D, k_delta in darkness
    k_delta_L = r_c + k_d_L
    k_delta_D = r_c + k_d_D
    #-- k_a: association rate of free GI and ZTL, k_a = k_delta/K
    #-- k_a_L = k_delta_L/K_L, association rate of free GI and ZTL in light
    #-- k_a_D = k_delta_D/K_D, association rate of free GI and ZTL in darkness
    k_a_L = k_delta_L/K_L
    k_a_D = k_delta_D/K_D
    #-- parameter values for the system of ODEs
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = k_a_D
    # pv[5] = k_delta_L
    # pv[6] = k_a_L - k_a_D
    # pv[7] = k_delta_D - k_delta_L
    pv = [GI_sp,g_A,r_A - r_c,r_c,k_a_D,k_delta_L,k_a_L - k_a_D,k_delta_D - k_delta_L]
    #-- solve the system of ODEs
    t_eval = numpy.linspace(0.0,t_end,int(t_end/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_GI_ZTL_full(t,y,pv),\
                    t_span=[t_start,t_end],y0=[ZTL_y_data[0]*w_ZTL,0.0],method='LSODA',\
                    t_eval=t_eval,max_step=dt)
    ZTL_sol = sol.y[0]/w_ZTL
    #-- return
    return t_eval,splev(t_eval,GI_sp,der=0),splev(t_eval,ZTL_sp,der=0),ZTL_sol
###

###
def GI_ZTL_gamma(t_start,t_end,dt,ZTL_data_file,GI_data_file,
                 w_ZTL,w_GI,g_A,r_A,r_c,k_d_L,k_d_D,K_L,K_D):
    #-- input
    #-- ODE solver parameters: t_start,t_end,dt
    #-- normalized protein level of ZTL: normalized_ZTL_data.csv
    #-- normalized protein level of GI : normalized_GI_data.csv
    #-- w_ZTL : scaling coefficient of ZTL, unit = nM
    #-- w_GI  : scaling coefficient of GI, unit = nM
    #-- g_A   : ZTL synthesis rate
    #-- r_A   : degradation rate of free ZTL
    #-- r_c   : degradation rate of GI-binding ZTL
    #-- k_d_L : dissociation rate of GI-binding ZTL in light
    #-- k_d_D : dissociation rate of GI-binding ZTL in darkness
    #-- K_L   : the Michaelis constant in light
    #-- K_D   : the Michaelis constant in darkness
    #
    #-- output
    #-- normalized protein levels of GI at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated using C_gamma
    #
    #-- Read data file, normalized protein level of ZTL
    ZTL_sp = protein_level_spline(ZTL_data_file,'time','ZTL_norm',1.0,3)
    ZTL_data = pandas.read_csv(ZTL_data_file)
    ZTL_y_data = ZTL_data['ZTL_norm'].values
    #-- Read data file, normalized protein level of GI
    GI_sp = protein_level_spline(GI_data_file,'time','GI_norm',w_GI,3)
    #-- k_delta = r_c + k_d
    #-- k_delta_L = r_c + k_d_L, k_delta in light
    #-- k_delta_D = r_c + k_d_D, k_delta in darkness
    k_delta_L = r_c + k_d_L
    k_delta_D = r_c + k_d_D
    #-- parameter values for the ODE
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = K_L
    # pv[5] = K_D - K_L
    # pv[6] = k_delta_L
    # pv[7] = k_delta_D - k_delta_L
    pv = [GI_sp,g_A,r_A - r_c,r_c,K_L,K_D - K_L,k_delta_L,k_delta_D - k_delta_L]
    #-- solve the system of ODEs
    t_eval = numpy.linspace(t_start,t_end,int(t_end/dt) + 1)
    sol = ddeint(func=lambda A_func,t : ODE_GI_ZTL_gamma(A_func,t,pv),
                 g=lambda t : ZTL_y_data[0]*w_ZTL,tt=t_eval)
    ZTL_sol = sol.astype(float)/w_ZTL
    #-- return
    return t_eval,splev(t_eval,GI_sp,der=0),splev(t_eval,ZTL_sp,der=0),ZTL_sol
###

###
def GI_ZTL_tQSSA(t_start,t_end,dt,ZTL_data_file,GI_data_file,
                 w_ZTL,w_GI,g_A,r_A,r_c,K_L,K_D):
    #-- input
    #-- ODE solver parameters: t_start,t_end,dt
    #-- normalized protein level of ZTL: normalized_ZTL_data.csv
    #-- normalized protein level of GI : normalized_GI_data.csv
    #-- w_ZTL : scaling coefficient of ZTL, unit = nM
    #-- w_GI  : scaling coefficient of GI, unit = nM
    #-- g_A   : ZTL synthesis rate
    #-- r_A   : degradation rate of free ZTL
    #-- r_c   : degradation rate of GI-binding ZTL
    #-- K_L   : the Michaelis constant in light
    #-- K_D   : the Michaelis constant in darkness
    #
    #-- output
    #-- normalized protein levels of GI at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated using spline representation
    #-- normalized protein levels of ZTL at time t evaluated using C_tQSSA
    #
    #-- Read data file, normalized protein level of ZTL
    ZTL_sp = protein_level_spline(ZTL_data_file,'time','ZTL_norm',1.0,3)
    ZTL_data = pandas.read_csv(ZTL_data_file)
    ZTL_y_data = ZTL_data['ZTL_norm'].values
    #-- Read data file, normalized protein level of GI
    GI_sp = protein_level_spline(GI_data_file,'time','GI_norm',w_GI,3)
    #-- parameter values for the system of ODEs
    # pv[0] = B(t)
    # pv[1] = g_A
    # pv[2] = r_A - r_c
    # pv[3] = r_c
    # pv[4] = K_L
    # pv[5] = K_D - k_L
    pv = [GI_sp,g_A,r_A - r_c,r_c,K_L,K_D - K_L]
    #-- solve the system of ODEs
    t_eval = numpy.linspace(t_start,t_end,int(t_end/dt) + 1)
    sol = solve_ivp(fun=lambda t,y : ODE_GI_ZTL_tQSSA(t,y,pv),\
                    t_span=[t_start,t_end],y0=[ZTL_y_data[0]*w_ZTL,0.0],method='LSODA',\
                    t_eval=t_eval,max_step=dt)
    ZTL_sol = sol.y[0]/w_ZTL
    #-- return
    return t_eval,splev(t_eval,GI_sp,der=0),splev(t_eval,ZTL_sp,der=0),ZTL_sol
###

###
def main_B_protein_protein_interaction_GI_ZTL():
    #-- input
    #-- normalized protein level of ZTL: normalized_ZTL_data.csv
    #-- normalized protein level of GI : normalized_GI_data.csv
    #-- w_ZTL : scaling coefficient of ZTL, unit = nM
    #-- w_GI  : scaling coefficient of GI, unit = nM
    #-- g_A   : ZTL synthesis rate
    #-- r_A   : degradation rate of free ZTL
    #-- r_c   : degradation rate of GI-binding ZTL
    #-- k_d_L : dissociation rate of GI-binding ZTL in light
    #-- k_d_D : dissociation rate of GI-binding ZTL in darkness
    #-- K_L   : the Michaelis constant in light
    #-- K_D   : the Michaelis constant in darkness
    ZTL_data_file = 'normalized_ZTL_data.csv'
    GI_data_file = 'normalized_GI_data.csv'
    w_ZTL = 3.040866733
    w_GI = 6.874374585
    g_A = 2.167419787
    r_A = 9.233286601
    r_c = 0.106102948
    k_d_L = 0.159042633
    k_d_D = 0.164336053
    K_L = 0.224852451
    K_D = 0.243143378
    #-- ODE solver parameters
    t_start = 0.0
    t_end = 20.0*24.0
    dt = 0.05
    #-- Case 1, full equaions
    t_eval,GI_sp,ZTL_sp,ZTL_full = GI_ZTL_full(t_start,t_end,dt,ZTL_data_file,GI_data_file,w_ZTL,w_GI,g_A,r_A,r_c,k_d_L,k_d_D,K_L,K_D)
    #-- Case 2, C(t) ~ C_gamma(t)
    _,_,_,ZTL_gamma = GI_ZTL_gamma(t_start,t_end,dt,ZTL_data_file,GI_data_file,w_ZTL,w_GI,g_A,r_A,r_c,k_d_L,k_d_D,K_L,K_D)
    #-- Case 3, C(t) ~ C_tQSSA(t)
    _,_,_,ZTL_tQSSA = GI_ZTL_tQSSA(t_start,t_end,dt,ZTL_data_file,GI_data_file,w_ZTL,w_GI,g_A,r_A,r_c,K_L,K_D)
    #-- save the solution
    df_sol = pandas.DataFrame({})
    df_sol['t'] = t_eval
    df_sol['GI'] = GI_sp
    df_sol['ZTL'] = ZTL_sp
    df_sol['ZTL_full'] = ZTL_full
    df_sol['ZTL_gamma'] = ZTL_gamma
    df_sol['ZTL_tQSSA'] = ZTL_tQSSA
    df_sol.to_csv('B_protein_protein_interaction_GI_ZTL.csv',index=False)
    #-- return
    return 0
###

####-- END Functions

####-- Main script

main_B_protein_protein_interaction_GI_ZTL()

####-- END ---####