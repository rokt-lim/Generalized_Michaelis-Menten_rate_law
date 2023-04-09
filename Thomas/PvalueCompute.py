# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:23 2021

@author: Thomas L. P. Martin
"""

import numpy as np
import scipy as sc
from random import randint
from random import shuffle

PhsQSSA = np.load('PhasesQSSA.npy')
PhtQSSA = np.load('PhasetQSSA.npy')
Phv1 = np.load('Phasev1.npy')

#This function returns the difference between the median of absolute phase shifts of CtQ and the median absolute phase shifts of Cgamma 
def P_valuePhase_tQSSA_v1(Number):
    MedtQSSA = np.median(abs(PhtQSSA)) - np.median(abs(Phv1))
    P_value = 0
    for i in range(Number):
        #For each iteration, we create two fake lists of phase shifts
        Fakev1 = []
        FaketQSSA = []
        for j in range(len(Phv1)):
            #For each value of phase shift, we switch the values between tQ and gamma with probability 0.5 or keep the same values
            rand = randint(0, 1)
            if rand == 0:
                Fakev1.append(Phv1[j]) 
                FaketQSSA.append(PhtQSSA[j])
            elif rand == 1:
                Fakev1.append(PhtQSSA[j]) 
                FaketQSSA.append(Phv1[j])
            #We compute the difference of median of absolute values of the two fake lists, if this difference is greater than our first value, we increment the p value by 1
            if (np.median(abs(np.array(FaketQSSA))) - np.median(abs(np.array(Fakev1)))) >= MedtQSSA:
                    P_value += 1
    return MedtQSSA, P_value / Number

#This function returns the difference between the median of absolute phase shifts of CtQ and the median absolute phase shifts of Cgamma 
def P_valuePhase_sQSSA_v1(Number):
    MedsQSSA = np.median(abs(PhsQSSA)) - np.median(abs(Phv1))
    P_value = 0
    for i in range(Number):
        #For each iteration, we create two fake lists of phase shifts
        Fakev1 = []
        FakesQSSA = []
        for j in range(len(Phv1)):
            #For each value of phase shift, we switch the values between tQ and gamma with probability 0.5 or keep the same values
            rand = randint(0, 1)
            if rand == 0:
                Fakev1.append(Phv1[j]) 
                FakesQSSA.append(PhsQSSA[j])
            elif rand == 1:
                Fakev1.append(PhsQSSA[j]) 
                FakesQSSA.append(Phv1[j])
            #We compute the difference of median of absolute values of the two fake lists, if this difference is greater than our first value, we increment the p value by 1
            if (np.median(abs(np.array(FakesQSSA))) - np.median(abs(np.array(Fakev1)))) >= MedsQSSA:
                    P_value += 1
    return MedsQSSA, P_value / Number
    


        
PhtQSSA_CondA = np.load('PhasetQSSA_CondA.npy')
PhsQSSA_CondA = np.load('PhasesQSSAA.npy')
Phv1_CondA = np.load('Phasev1_CondA.npy')

#This function reports the number of cases when the absolute phase shift of Cgamma is smaller than both absolute phase shift of CtQ and CsQ by at least 1 hour
#As previously, the p value is computing using random permutation between the three phase shifts list and computing the number of cases when the absolute phase shift of Cgamma is smaller than both absolute phase shift of CtQ and CsQ by at least 1 hour, if this number is greater than our first value, p value is incremented by 1
def P_value_PhaseCondA_1hour(Number):
    P_value = 0
    Number1hour = 0
    for i in range(len(Phv1_CondA)):
        if min(PhtQSSA_CondA[i] - Phv1_CondA[i], PhsQSSA_CondA[i] - Phv1_CondA[i]) > 1:
            Number1hour += 1
    for j in range(Number):
        FakePhtQSSACondA = []
        FakePhsQSSACondA = []
        FakePhv1CondA = []
        FakeNumber1hour = 0
        for k in range(len(Phv1_CondA)):
            L = [Phv1_CondA[k], PhtQSSA_CondA[k], PhsQSSA_CondA[k]]
            shuffle(L)
            FakePhtQSSACondA.append(L[1])
            FakePhsQSSACondA.append(L[2])
            FakePhv1CondA.append(L[0])
            if min(FakePhtQSSACondA[j] - FakePhv1CondA[j], FakePhsQSSACondA[j] - FakePhv1CondA[j]) > 1:
                FakeNumber1hour += 1
        if FakeNumber1hour >= Number1hour:
            P_value += 1
    return Number1hour / len(Phv1_CondA), P_value / Number

#This function reports the number of cases when the absolute phase shift of Cgamma is smaller than both absolute phase shift of CtQ and CsQ by at least 2 hours
#As previously, the p value is computing using random permutation between the three phase shifts list and computing the number of cases when the absolute phase shift of Cgamma is smaller than both absolute phase shift of CtQ and CsQ by at least 2 hours, if this number is greater than our first value, p value is incremented by 1
def P_value_PhaseCondA_2hour(Number):
    P_value = 0
    Number2hour = 0
    for i in range(len(Phv1_CondA)):
        if min(PhtQSSA_CondA[i] - Phv1_CondA[i], PhsQSSA_CondA[i] - Phv1_CondA[i]) > 2:
            Number2hour += 1
    for j in range(Number):
        FakePhtQSSACondA = []
        FakePhsQSSACondA = []
        FakePhv1CondA = []
        FakeNumber2hour = 0
        for k in range(len(Phv1_CondA)):
            L = [Phv1_CondA[k], PhtQSSA_CondA[k], PhsQSSA_CondA[k]]
            shuffle(L)
            FakePhtQSSACondA.append(L[1])
            FakePhsQSSACondA.append(L[2])
            FakePhv1CondA.append(L[0])
            if min(FakePhtQSSACondA[j] - FakePhv1CondA[j], FakePhsQSSACondA[j] - FakePhv1CondA[j]) > 2:
                FakeNumber2hour += 1
        if FakeNumber2hour >= Number2hour:
            P_value += 1
    return Number2hour / len(Phv1_CondA), P_value / Number

Simv1 = np.load('Simv1.npy') 
SimtQSSA = np.load('SimtQSSA.npy') 
SimsQSSA = np.load('SimsQSSA.npy')

#This function computes the Spearman correlation coefficient between the similarity after phase correction with Cfull of Cgamma and CtQ
#To compute the p value, at each iteration we swap randomly the list of similarities of Cgamma, compute the Spearman correlation coefficient of this new list with the similarities of CtQ, if this new coefficient if greater than our first one, the p value is incremented by 1
def P_value_Spearman_Sim_tQSSA_v1(Number):
    P_value = 0
    Corr = sc.stats.spearmanr(Simv1, SimtQSSA)[0]
    for i in range(Number):
        FakeSimv1 = np.copy(Simv1)
        FakeSimtQSSA = np.copy(SimtQSSA)
        np.random.shuffle(FakeSimv1)
        FakeCorr = sc.stats.spearmanr(FakeSimv1, FakeSimtQSSA)[0]
        if FakeCorr >= Corr:
            P_value += 1
    return Corr, P_value / Number

#This function computes the difference between the median of similarities of Cgamma and the similarities of CsQ
#The p value is computed the same way as for the differences in phase shift
def P_value_Sim_v1_sQSSA(Number):
    P_value = 0
    Medv1sQSSA = np.median(Simv1) - np.median(SimsQSSA)
    for i in range(Number):
        FakeSimv1 = []
        FakeSimsQSSA = []
        for j in range(len(Simv1)):
            rand = randint(0, 1)
            if rand == 0:
                FakeSimv1.append(Simv1[j]) 
                FakeSimsQSSA.append(SimsQSSA[j])
            elif rand == 1:
                FakeSimv1.append(SimsQSSA[j]) 
                FakeSimsQSSA.append(Simv1[j])
            if (np.median(np.array(FakeSimv1)) - np.median(np.array(FakeSimsQSSA))) >= Medv1sQSSA:
                P_value += 1
    return Medv1sQSSA, P_value / Number

#This function computes the difference between the median of similarities of Cgamma and the similarities of CtQ
#The p value is computed the same way as for the differences in phase shift
def P_value_Sim_tQSSA_sQSSA(Number):
    P_value = 0
    MedtQSSAsQSSA = np.median(SimtQSSA) - np.median(SimsQSSA)
    for i in range(Number):
        FakeSimtQSSA = []
        FakeSimsQSSA = []
        for j in range(len(Simv1)):
            rand = randint(0, 1)
            if rand == 0:
                FakeSimtQSSA.append(SimtQSSA[j]) 
                FakeSimsQSSA.append(SimsQSSA[j])
            elif rand == 1:
                FakeSimtQSSA.append(SimsQSSA[j]) 
                FakeSimsQSSA.append(SimtQSSA[j])
            if (np.median(np.array(FakeSimtQSSA)) - np.median(np.array(FakeSimsQSSA))) >= MedtQSSAsQSSA:
                P_value += 1
    return MedtQSSAsQSSA, P_value / Number

Eps_tQSSA = np.load('epsilon_tQ_Table')
Eps_v1 = np.load('epsilon_v1_Table')

EpstQ = []
Epsgamma = []

for i in range(len(Eps_tQSSA)):
    EpstQ.append(np.max(abs(Eps_tQSSA[i])))
    Epsgamma.append(np.max(abs(Eps_v1[i])))

#This function computes the Spearman correlation coefficient between the max of epsilon gamma on one period and the absolute phase shift of Cgamma
#The p value is computed the way as for the Spearman correlation coefficient of similarities
def P_value_Spearman_phasev1_Epsgamma(Number):
    P_value = 0
    Corr = sc.stats.spearmanr(abs(Phv1), Epsgamma)[0]
    for i in range(Number):
        FakePhv1 = np.copy(Phv1)
        np.random.shuffle(FakePhv1)
        FakeCorr = sc.stats.spearmanr(abs(FakePhv1), Epsgamma)[0]
        if FakeCorr >= Corr:
            P_value += 1
    return Corr, P_value / Number

#This function computes the Spearman correlation coefficient between the max of epsilon tQ on one period and the absolute phase shift of CtQ
#The p value is computed the way as for the Spearman correlation coefficient of similarities
def P_value_Spearman_phasetQSSA_EpstQ(Number):
    P_value = 0
    Corr = sc.stats.spearmanr(abs(PhtQSSA), EpstQ)[0]
    for i in range(Number):
        FakePhtQ = np.copy(PhtQSSA)
        np.random.shuffle(FakePhtQ)
        FakeCorr = sc.stats.spearmanr(abs(FakePhtQ), EpstQ)[0]
        if FakeCorr >= Corr:
            P_value += 1
    return Corr, P_value / Number
       

