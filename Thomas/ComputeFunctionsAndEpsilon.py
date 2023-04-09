# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:46:42 2021

@author: Thomas L. P. Martin
"""


#This Python file can be used to generate functions A_bar, B_bar, mu_A, mu_B and their derivatives in the case Protein-Protein interaction
#It also generates the full solution C_bar of the differential equation (2) and the approximators C_gamma (C_v1 in this code), C_gamma_p, C_tQSSA and C_sQSSA as well as the error functions epsilon1, epsilon2, epsilon_gamma and epsilontQ.

import math as math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from random import uniform

#Functions used for the simulation

#A_bar
def A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return 0.5 * alpha_A * A_max * math.sin(2*math.pi*t / (T) - math.pi/2) + (1 - 0.5 * alpha_A) * A_max

#B_bar
def B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return 0.5 * alpha_B * B_max * math.sin(2*math.pi*t / (T) - math.pi/2 - phi_B) + (1 - 0.5 * alpha_B) * B_max

#mu_A
def mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return math.pi * alpha_A * A_max *math.cos(2 * math.pi * t / (T) - math.pi / 2) / (k_delta * T * A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T))

#Derivative of mu_A
def d_mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return - math.pi * alpha_A * A_max * ( 2 * math.pi * math.sin(2 * math.pi * t / T - math.pi / 2) + mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) * math.cos(2 * math.pi * t / T - math.pi / 2)) / (T * k_delta * A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T))

#mu_B
def mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return math.pi * alpha_B * B_max *math.cos(2 * math.pi * t / (T) - math.pi / 2 - phi_B) / (k_delta * T * B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T))

#Derivative of mu_B
def d_mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return - math.pi * alpha_B * B_max * ( 2 * math.pi * math.sin(2 * math.pi * t / T - math.pi / 2 - phi_B) + mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) * math.cos(2 * math.pi * t / T - math.pi / 2 - phi_B)) / (T * k_delta * B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T))

#CsQSSA
def C_sQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    C_star_t = A_t * B_t / (1 + A_t + B_t)
    return C_star_t

#CtQSSA
def C_tQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    C_star_t = 0.5 * (1 + A_t + B_t - math.sqrt((1 + A_t + B_t)**2 - 4 * A_t * B_t))
    return C_star_t

#C_gamma
def Cv1(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
    if delta_square > 0:
        Cv1 = min(C_tQSSA(t - 1 / (k_delta * math.sqrt(delta_square)), A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T), min(A_t, B_t))
    else:
        Cv1 = 0
    return Cv1

#Derivative of C_bar in the ODE (2)
def derC(t, C, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return k_delta * (A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) - C) * (B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) - C) - k_delta * C

#Epsilon1 from the equation (20)
def epsilon1(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muA_t = mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muB_t = mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    
    C_tQSSA = 0.5 * (1 + A_t + B_t - math.sqrt((1 + A_t + B_t)**2 - 4 * A_t * B_t))
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
    
    d_CtQSSA = ((B_t - C_tQSSA) * A_t * muA_t + (A_t - C_tQSSA) * B_t * muB_t) / math.sqrt(delta_square)
    
    epsilon1 = abs(d_CtQSSA) / delta_square
    return epsilon1

#Epsilon2 from the equation (22)
def epsilon2(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muA_t = mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muB_t = mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
    
    epsilon2 = abs((1 + A_t - B_t) * A_t * muA_t + (1 + B_t - A_t) * B_t * muB_t) / (math.sqrt(delta_square) * delta_square)
    return epsilon2

#Epsilon_gamma from the equation (23)
def epsilon_gamma(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muA_t = mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muB_t = mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    d_muA_t = d_mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    d_muB_t = d_mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    
    CtQSSA = 0.5 * (1 + A_t + B_t - math.sqrt((1 + A_t + B_t)**2 - 4 * A_t * B_t))
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
        
    d2_CtQSSA = ((delta_square - 2*A_t * (1 + A_t - CtQSSA)) * (B_t - CtQSSA) * A_t * muA_t**2 + (delta_square - 2*B_t * (1 + B_t - CtQSSA)) * (A_t - CtQSSA) * B_t * muB_t**2 + (1 + delta_square - (A_t - B_t)**2) * A_t * B_t * muA_t * muB_t + delta_square * ((B_t - CtQSSA) * A_t * d_muA_t + (A_t - CtQSSA) * B_t * d_muB_t)) / (delta_square * math.sqrt(delta_square))
    
    epsilon_gamma = abs(d2_CtQSSA) / (2 * delta_square * C_tQSSA(t - math.sqrt(delta_square), A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T))
    return epsilon_gamma

#Epsilon_tQ from the equation (25)
def epsilon_tQ(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    A_t = A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    B_t = B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muA_t = mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    muB_t = mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    d_muA_t = d_mu_A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    d_muB_t = d_mu_B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T)
    
    C_tQSSA = 0.5 * (1 + A_t + B_t - math.sqrt((1 + A_t + B_t)**2 - 4 * A_t * B_t))
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
    
    d_CtQSSA = ((B_t - C_tQSSA) * A_t * muA_t + (A_t - C_tQSSA) * B_t * muB_t) / math.sqrt(delta_square)
    d2_CtQSSA = ((delta_square - 2*A_t * (1 + A_t - C_tQSSA)) * (B_t - C_tQSSA) * A_t * muA_t**2 + (delta_square - 2*B_t * (1 + B_t - C_tQSSA)) * (A_t - C_tQSSA) * B_t * muB_t**2 + (1 + delta_square - (A_t - B_t)**2) * A_t * B_t * muA_t * muB_t + delta_square * ((B_t - C_tQSSA) * A_t * d_muA_t + (A_t - C_tQSSA) * B_t * d_muB_t)) / (delta_square * math.sqrt(delta_square))
    
    epsilon_tQ = abs(d_CtQSSA / math.sqrt(delta_square) - d2_CtQSSA / delta_square) / C_tQSSA
    return epsilon_tQ

#Function to generate N random sets of parameters, compute function A_bar, B_bar, C_full, C_gamma, C_tQSSA, C_sQSSA, epsilon1, epsilon2, epsilon_gamma and epsilon_tQ
# 0.01 <= A_bar_max <= 100
# 0.01 <= B_bar_max <= 100
# 0.1 <= k_delta <= 10
# 0 <= alpha_A < 1    
# 0 <= alpha_B <= 1
# 0 <= phi_B <= pi
#Parameters of this function are N the number of simulations, t0 and tf the boundaries of the computed functions and results, t0eval and tfeval the boundaries to compute Cfull, dt the time step and T the period of A_bar and B_bar.
def RandomGenerate(N, t0, tf, t0eval, tfeval, dt, T):
    
    #Interval for the solving of Cbar and compute the functions
    t_eval = [i*0.05 for i in range(int(t0eval/dt), int(tfeval/dt) + 1)] 
    Time_list = [i*0.05 for i in range(int(t0/dt), int(tf/dt) + 1)]
    
    #Lists to store the functions
    List_Parameter = []

    A_Table = []

    B_Table = []

    C_full_Table = []

    C_tQSSA_Table = []
    
    C_sQSSA_Table = []

    Cv1_Table = []
    
    epsilon1_Table = []
    
    epsilon2_Table = []
    
    epsilon_tQ_Table = []
    
    epsilon_gamma_Table = []
    
    for i in range(N):
        #Generation of the random set of parameters
        A_max = 10**uniform(-2, 2)
        B_max = 10**uniform(-2, 2)
        k_delta = 10**uniform(-1, 1)
        alpha_A = uniform(0, 0.99999999)
        alpha_B = uniform(0, 0.99999999)
        phi_B = uniform(0, math.pi)
        
        
        #Computation of the full solution Cbar
        C_full = sc.integrate.solve_ivp(lambda t, C: derC(t, C, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T), [t0eval, tfeval], [0], max_step = 0.05, t_eval = t_eval, method = 'LSODA')
        C_full_time = C_full.y[0][int(t0/dt) : int(tf/dt) + 1]
        
        #Test if the function is relevant for our case with the use of alpha = (max(Cbar) - min(Cbar)) / max(Cbar), if alpha < 0.2, the solution do not oscillate enought to be relevant for our case
        #We compute this alpha value on an interval of size T long time after the time 0 to let enought time for the solution to reach its oscillatory behaviour
        alpha = (np.max(C_full_time) - np.min(C_full_time)) / np.max(C_full_time)
        if alpha >= 0.2:
            C_full_Table.append(np.copy(C_full.y[0][int(t0/dt) : int(tf/dt) + 1]))
            
            #Computation and storage of all the functions and parameters
            List_Parameter.append(np.array([A_max, B_max, k_delta, alpha_A, alpha_B, phi_B]))
            
            A_Table.append(np.array([A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            B_Table.append(np.array([B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            
            C_tQSSA_Table.append(np.array([C_tQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            C_sQSSA_Table.append(np.array([C_sQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            
            Cv1_Table.append(np.array([Cv1(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            
            epsilon1_Table.append(np.array([epsilon1(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            epsilon2_Table.append(np.array([epsilon2(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            epsilon_tQ_Table.append(np.array([epsilon_tQ(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            epsilon_gamma_Table.append(np.array([epsilon_gamma(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list]))
            
    #Save the list of functions and parameters in npy files
    np.save('List_Parameter.npy', List_Parameter)

    np.save('A_Table.npy', A_Table)
    np.save('B_Table.npy', B_Table)

    np.save('C_full_Table.npy', C_full_Table)

    np.save('C_tQSSA_Table', C_tQSSA_Table)
    np.save('C_sQSSA_Table', C_sQSSA_Table)
    np.save('Cv1_Table', Cv1_Table)
    
    np.save('epsilon1_Table', epsilon1_Table)
    np.save('epsilon2_Table', epsilon2_Table)
    np.save('epsilon_tQ_Table', epsilon_tQ_Table)
    np.save('epsilon_gamma_Table', epsilon_gamma_Table)

 #This function draws A, B, Cfull, Cgamma, Cgamma_p, CtQ and CsQ for a particular set of parameters  
def Draw_Individual(A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T, dt, t0, tf, t0eval, tfeval):
    t_eval = [i*0.05 for i in range(int(t0eval/dt), int(tfeval/dt) + 1)]
    C_sol_full = sc.integrate.solve_ivp(lambda t, C: derC(t, C, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T), [t0eval, tfeval], [0], max_step = 0.01, t_eval = t_eval)
    low_bound = int(t0/dt)
    high_bound = int(tf/dt)

    Time_list = C_sol_full.t[low_bound : high_bound + 1]
    C_sol = C_sol_full.y[0][low_bound : high_bound + 1]
            
    A_t = np.array([A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    B_t = np.array([B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
            
    C_tQSSA = np.array([C_tQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    C_sQSSA = np.array([C_sQSSA(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    
    C_tQ_mean = sc.integrate.trapz(C_tQSSA, dx=0.05)/(tf - t0)
    
    C_v1 = np.array([Cv1(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    
    plt.plot(Time_list, A_t, color = 'orange', label = 'A_bar(t)')
    plt.plot(Time_list, B_t, color = 'darkcyan', label = 'B_bar(t)')
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.ylim(0, 1.1 * max(np.max(A_t), np.max(B_t)))
    plt.xlabel('Time (hour)')
    plt.title('Draw Full solution, A_bar and B_bar')
    plt.legend()
    plt.show()    
    
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.plot(Time_list, C_sQSSA, color = 'blue', label = 'sQSSA')
    plt.plot(Time_list, C_tQSSA, color = 'red', label = 'tQSSA')
    plt.xlabel('Time (hour)')
    plt.ylim(0, 1.1 * max(np.max(C_sol), np.max(C_tQSSA), np.max(C_sQSSA)))
    plt.title('Comparison Time serie Full solution, tQSSA and sQSSA')
    plt.legend()
    plt.show()
    
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.plot(Time_list, C_v1, color = 'purple', label = 'Cv1')
    plt.ylim(0, 1.1 * max(np.max(C_sol), np.max(C_v1)))
    plt.xlabel('Time (hour)')
    plt.title('Comparison Time serie Full solution, v1 and gamma p')
    plt.legend()
    plt.show()
        
#Computation of the random case
#This part of the code is about the irregular case in equation (46) 

#The number of functions used in A_bar and B_bar        
N = 10
        
def A_t(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_A, phi_B, T):
    return 0.5 * alpha_A * A_max * math.sin(2*math.pi*t / (T) - math.pi/2 - phi_A) + (1 - 0.5 * alpha_A) * A_max

def B_t(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_A, phi_B, T):
    return 0.5 * alpha_B * B_max * math.sin(2*math.pi*t / (T) - math.pi/2 - phi_B) + (1 - 0.5 * alpha_B) * B_max

def A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    Aserie = 0
    for i in range(N):
        Aserie += A(t, L_A_max[i], L_B_max[i], k_delta, L_alpha_A[i], L_alpha_B[i], L_phi_A[i], L_phi_B[i], L_TA[i])
    return Aserie / N

def B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    Bserie = 0
    for i in range(N):
        Bserie += B(t, L_A_max[i], L_B_max[i], k_delta, L_alpha_A[i], L_alpha_B[i], L_phi_A[i], L_phi_B[i], L_TB[i])
    return Bserie / N

def CN_tQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    A_t = A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    B_t = B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    C_star_t = 0.5 * (1 + A_t + B_t - math.sqrt((1 + A_t + B_t)**2 - 4 * A_t * B_t))
    return C_star_t

def CN_sQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    A_t = A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    B_t = B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    C_star_t = A_t * B_t / (1 + A_t + B_t)
    return C_star_t

def CNv1(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    A_t = A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    B_t = B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB)
    delta_square = (1 + A_t + B_t)**2 - 4 * A_t * B_t
    if delta_square > 0:
        Cv1 = min(CN_tQSSA(t - 1 / (k_delta * math.sqrt(delta_square)), L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB), min(A_t, B_t))
    else:
        Cv1 = 0
    return Cv1    

def derCN(t, C, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB):
    return k_delta * (A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) - C) * (B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) - C) - k_delta * C   

#Function to generate N random sets of parameters, compute function AN_bar, BN_bar, CN_full, CN_gamma, CN_tQSSA, CN_sQSSA
# 0.1 <= A_bar_max i <= 10
# 0.1 <= B_bar_max i <= 10
# 0.1 <= k_delta <= 10
# 0.5 <= alpha_A < 1    
# 0.5 <= alpha_B <= 1
# -pi <= phi_A <= pi
# -pi <= phi_B <= pi
# 10 <= TA <= 40
# 10 <= TB <= 40    
#Parameters of this function are N the number of simulations, t0 and tf the boundaries of the computed functions and results, t0eval and tfeval the boundaries to compute Cfull and dt the time step   
def RandomGenerate_Irregular(N, t0, tf, t0eval, tfeval, dt):
    
    #Interval for the solving of Cbar and compute the functions
    t_eval = [i*0.05 for i in range(int(t0eval/dt), int(tfeval/dt) + 1)] 
    Time_list = [i*0.05 for i in range(int(t0/dt), int(tf/dt) + 1)]
    
    for i in range(N):
        L_A_max = np.array([10**uniform(-1, 1),10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1)]) 
        L_B_max = np.array([10**uniform(-1, 1),10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1), 10**uniform(-1, 1)])
        k_delta = 10**uniform(-1, 1)
        L_alpha_A = np.random.uniform(0.5, 0.99999, 10)
        L_alpha_B = np.random.uniform(0.5, 0.99999, 10)
        L_phi_A = np.random.uniform(-math.pi + 0.000001, math.pi, 10)
        L_phi_B = np.random.uniform(-math.pi + 0.000001, math.pi, 10)
        L_TA = np.random.uniform(10, 40, 10)
        L_TB = np.random.uniform(10, 40, 10)
        
        print(L_A_max)
        print(L_B_max)
        print(k_delta)
        print(L_alpha_A)
        print(L_alpha_B)
        print(L_phi_A)
        print(L_phi_B)
        print(L_TA)
        print(L_TB)
        
        
        #Computation of the full solution Cbar
        CN_full = sc.integrate.solve_ivp(lambda t, C: derCN(t, C, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB), [t0eval, tfeval], [0], max_step = 0.05, t_eval = t_eval, method = 'LSODA')
        

        CN_full_t = np.copy(CN_full.y[0][int(t0/dt) : int(tf/dt) + 1])
            
        #Computation and storage of all the functions and parameters
            
        AN_t = np.array([A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
        BN_t = np.array([B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
            
        CN_tQSSA_t = np.array([CN_tQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
        CN_sQSSA_t = np.array([CN_sQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
        CNv1_t = np.array([CNv1(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
            
    #Save the list of functions and parameters in npy files
    plt.plot(Time_list, AN_t, color = 'orange', label = 'A_bar(t)')
    plt.plot(Time_list, BN_t, color = 'darkcyan', label = 'B_bar(t)')
    plt.plot(Time_list , CN_full_t, color = 'black', label = 'Full solution')
    plt.ylim(0, 1.1 * max(np.max(AN_t), np.max(BN_t)))
    plt.xlabel('Time (hour)')
    plt.title('Draw Full solution, A_bar and B_bar')
    plt.legend()
    plt.show()
    
    
    plt.plot(Time_list , CN_full_t, color = 'black', label = 'Full solution')
    plt.plot(Time_list , CN_sQSSA_t, color = 'blue', label = 'sQSSA')
    plt.plot(Time_list , CN_tQSSA_t, color = 'red', label = 'tQSSA')
    plt.xlabel('Time (hour)')
    plt.ylim(0, 1.1 * max(np.max(CN_full_t), np.max(CN_tQSSA_t), np.max(CN_sQSSA_t)))
    plt.title('Comparison Time serie Full solution, tQSSA and sQSSA')
    plt.legend()
    plt.show()
    
    plt.plot(Time_list , CN_full, color = 'black', label = 'Full solution')
    plt.plot(Time_list , CNv1_t, color = 'purple', label = 'Cv1')
    plt.ylim(0, 1.1 * max(np.max(CN_full_t), np.max(CNv1_t)))
    plt.xlabel('Time (hour)')
    plt.title('Comparison Time serie Full solution and v1')
    plt.legend()
    plt.show()

#This function draws AN, BN, Cfull, Cgamma, CtQ and CsQ for a particular set of parameters
def Draw_Irregular(L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB, dt, t0, tf, t0eval, tfeval):
    t_eval = [i*0.05 for i in range(int(t0eval/dt), int(tfeval/dt) + 1)]
    C_sol_full = sc.integrate.solve_ivp(lambda t, C: derCN(t, C, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB), [t0eval, tfeval], [0], max_step = 0.01, t_eval = t_eval)
    low_bound = int(t0/dt)
    high_bound = int(tf/dt)

    Time_list = C_sol_full.t[low_bound : high_bound + 1]
    C_sol = C_sol_full.y[0][low_bound : high_bound + 1]
            
    A_t = np.array([A_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
    B_t = np.array([B_N(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
            
    C_tQSSA = np.array([CN_tQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
    C_sQSSA = np.array([CN_sQSSA(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
    C_v1 = np.array([CNv1(t, L_A_max, L_B_max, k_delta, L_alpha_A, L_alpha_B, L_phi_A, L_phi_B, L_TA, L_TB) for t in Time_list])
    
    plt.plot(Time_list, A_t, color = 'orange', label = 'A_bar(t)')
    plt.plot(Time_list, B_t, color = 'darkcyan', label = 'B_bar(t)')
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.ylim(0, 1.1 * max(np.max(A_t), np.max(B_t)))
    plt.xlabel('Time (hour)')
    plt.title('Draw Full solution, A_bar and B_bar')
    plt.legend()
    plt.show()    
    
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.plot(Time_list, C_sQSSA, color = 'blue', label = 'sQSSA')
    plt.plot(Time_list, C_tQSSA, color = 'red', label = 'tQSSA')
    plt.xlabel('Time (hour)')
    plt.ylim(0, 1.1 * max(np.max(C_sol), np.max(C_tQSSA), np.max(C_sQSSA)))
    plt.title('Comparison Time serie Full solution, tQSSA and sQSSA')
    plt.legend()
    plt.show()
    
    plt.plot(Time_list, C_sol, color = 'black', label = 'Full solution')
    plt.plot(Time_list, C_v1, color = 'purple', label = 'Cv1')
    plt.ylim(0, 1.1 * max(np.max(C_sol), np.max(C_v1)))
    plt.xlabel('Time (hour)')
    plt.title('Comparison Time serie Full solution and v1')
    plt.legend()
    plt.show()
    

