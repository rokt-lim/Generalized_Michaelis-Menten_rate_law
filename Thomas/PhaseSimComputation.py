# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:40:35 2021

@author: Thomas L. P. Martin
"""

import math as math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

ListParam = np.load('List_Parameter.npy')

Cfull = np.load('C_full_Table.npy')
CtQSSA = np.load('C_tQSSA_Table.npy')
CsQSSA = np.load('C_sQSSA_Table.npy')
Cv1 = np.load('C_v1_Table.npy')

dt = 0.05
t0 = 350
tf = 374
T = 24

#Quality tests of the functions
#The following functions are used to determine wether a function gives a good approximation of the full solution or not
    
#Phase Shift
#This function uses scipy function correlate to determine the phase shift between two signal 
def Phase(C1, C2, dt):
    #The two signals are repeated on 10 periods and average to zero to improve quality of the result
    C1periods = np.concatenate([C1 for i in range(10)]) - np.mean(C1)
    C2periods = np.concatenate([C2 for i in range(10)]) - np.mean(C2)
    shift = dt * (np.argmax(sc.signal.correlate(C1periods, C2periods, mode = 'same', method = 'fft')) - np.argmax(sc.signal.correlate(C1periods, C1periods, mode = 'same', method = 'fft')))
    #The result is in hour 
    return shift

#Similarity
#This function return a value between 0 and 1, the closer it is to 1, the better the approximation is 
def Similarity(C1, C2, dt):
    Simi_min = 0
    Simi_max = 0
    for i in range(len(C1)):
        Simi_min += dt * min(C1[i], C2[i])
        Simi_max += dt * max(C1[i], C2[i])
    if Simi_max ==0:
        return 1
    else:
        return Simi_min / Simi_max  
    
Phase_tQSSA = []
Phase_sQSSA = []
Phase_v1 = []

#This loop compute the phase shift of CtQ, CsQ and Cgamma compare to Cfull for all the parameter sets
for i in range(len(Cfull)):
    Phase_tQSSA.append(Phase(Cfull[i], CtQSSA[i], dt))
    Phase_sQSSA.append(Phase(Cfull[i], CsQSSA[i], dt))
    Phase_v1.append(Phase(Cfull[i], Cv1[i], dt))
    
np.save('PhasetQSSA.npy', Phase_tQSSA)
np.save('PhasesQSSA.npy', Phase_sQSSA)
np.save('Phasev1.npy', Phase_v1)
    
Sim_tQSSA = []
Sim_sQSSA = []
Sim_v1 = []

#A_bar
def A(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return 0.5 * alpha_A * A_max * math.sin(2*math.pi*t / (T) - math.pi/2) + (1 - 0.5 * alpha_A) * A_max

#B_bar
def B(t, A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T):
    return 0.5 * alpha_B * B_max * math.sin(2*math.pi*t / (T) - math.pi/2 - phi_B) + (1 - 0.5 * alpha_B) * B_max

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

#This loop compute the similarity of CtQ, CsQ and Cgamma compare to Cfull after correction of the phase shift for all the parameter sets
for i in range(len(Cfull)):
    (A_max, B_max, k_delta, alpha_A, alpha_B, phi_B) = ListParam[i]
    Time_list = [j*0.05 for j in range(int(t0/dt), int(tf/dt) + 1)]
    
    C_sQSSA_phased = np.array([C_sQSSA(t - Phase_sQSSA[i], A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    C_tQSSA_phased = np.array([C_tQSSA(t - Phase_tQSSA[i], A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    C_v1_phased = np.array([Cv1(t - Phase_v1[i], A_max, B_max, k_delta, alpha_A, alpha_B, phi_B, T) for t in Time_list])
    
    Sim_sQSSA.append(Similarity(Cfull[i], C_sQSSA_phased, dt))
    Sim_tQSSA.append(Similarity(Cfull[i], C_tQSSA_phased, dt))
    Sim_v1.append(Similarity(Cfull[i], C_v1_phased, dt))
    
np.save('SimtQSSA.npy', Sim_tQSSA)
np.save('SimsQSSA.npy', Sim_sQSSA)
np.save('Simv1.npy', Sim_v1)

#This function can be used to draw scatter plot between two lists
def Scatter_Plot(L1, L2, Name1, Name2, Title, xmin, xmax):
    plt.plot(L1, L2, 'o', markersize = 1, color = 'firebrick')
    plt.plot([xmin, xmax], [xmin, xmax], color = 'black')
    plt.title(Title)
    plt.xlabel(Name1)
    plt.ylabel(Name2)
    plt.legend()
    plt.show()
    
    
    
    