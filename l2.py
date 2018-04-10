#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:24:22 2018

@author: red
"""

import numpy as np
from scipy.integrate import dblquad,quad

ell_lens_max = 200
integral_ell_max = 2000

#file name
unlensed_spectra_filename = './data/test_scalCls.dat'
lens_potential_filename = './data/test_lenspotentialCls.dat'

unlensed_spectra = np.loadtxt(unlensed_spectra_filename)
lens_potential_spectra = np.loadtxt(lens_potential_filename)

#unlensed/lensed TT spectrum
ell = []
TT_unlensed = []
TT_lensed = []

#number of ell_max
ell_max = len(unlensed_spectra[:,0])+1
ell = unlensed_spectra[:,0]

TT_unlensed = unlensed_spectra[:,1]/unlensed_spectra[:,0]/(unlensed_spectra[:,0]+1)*2.0*np.pi
TT_lensed = lens_potential_spectra[:,1]/lens_potential_spectra[:,0]/(lens_potential_spectra[:,0]+1)*2.0*np.pi

#np.savetxt('try.dat', np.c_[ell,TT_lensed], fmt='%1.4e')

aL_min = 2
aL_max = ell_lens_max
Aa = np.zeros(aL_max-aL_min+1)
L_array = np.zeros(aL_max-aL_min+1)
dtheta = 0.06
dl1 = 1

for aL in range(aL_min,aL_max+1): #list of Aa[aL]
    L_array[aL-2] = aL
    for theta in np.arange(0,2*np.pi,dtheta): #\theta integral
        for l1 in range(2,integral_ell_max): #ell_1 integral
            l2 = int(np.sqrt(l1**2+aL**2-2*aL*l1*np.cos(theta))) #value of l2 from triangle relation
            s = (aL+l1+l2)/2.0
            if(max(aL,l1,l2)<s): #triangle inequality
                #if ((l1 >= aL - l2) & (l1 <= aL + l2) & (l2 < ell_max) & (l2 >= 2)): #triangle inequality
                tmp = TT_unlensed[l1-2]*(aL*l1*np.cos(theta)) + TT_unlensed[l2-2]*( (aL**2)-aL*l1*np.cos(theta) )
                tmp = tmp**2
                tmp = tmp/2.0/TT_lensed[l1-2]/TT_lensed[l2-2]
                tmp = tmp * l1 * dtheta * dl1/4/np.pi/np.pi
                Aa[aL-2] = Aa[aL-2] + tmp
    print aL
    Aa[aL-2] = aL*aL/Aa[aL-2]

np.savetxt('noise.dat', np.c_[L_array,Aa], fmt='%1.4e')



