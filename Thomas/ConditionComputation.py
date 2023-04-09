# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:39:44 2021

@author: Thomas L. P. Martin
"""


import numpy as np

Phasev1 = np.load('Phasev1.npy')
PhasetQSSA = np.load('PhasetQSSA.npy')
PhasesQSSA = np.load('PhasesQSSA.npy')

Phasev1_CondA = []
PhasetQSSA_CondA = []
PhasesQSSA_CondA = []

for i in range(len(Phasev1)):
    if abs(Phasev1[i]) >= 1 or abs(PhasetQSSA[i]) >= 1 or abs(PhasesQSSA[i]):
        Phasev1_CondA.append(Phasev1[i])
        PhasetQSSA_CondA.append(PhasetQSSA[i])
        PhasesQSSA_CondA.append(PhasesQSSA[i])