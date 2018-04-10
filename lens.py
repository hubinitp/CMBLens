#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:24:22 2018

@author: BH @ BNU
"""

import numpy as np
from scipy.integrate import dblquad,quad

ell_lens_max = 100
integral_ell_max = 3000

#noise params
#arcmin_to_radian
arcmin_to_radian = np.pi/180.0/60.0
#[\mu K]
T_CMB = 2.728E6
#Planck noise [\mu K*arcmin]
#D_T_arcmin = 27.0
#Almost perfect noise [\mu K*arcmin]
D_T_arcmin = 1.0
#Planck noise [\mu K*rad]
D_T = D_T_arcmin / arcmin_to_radian
#Planck FWHM of the beam [arcmin]
#fwmh_arcmin = 7.0
#Almost perfect FWHM of the beam [arcmin]
fwmh_arcmin = 4.0
fwmh = fwmh_arcmin * arcmin_to_radian


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

#noise spectrum
N_TT = np.zeros(ell_max-2+1)
for el in range(2,ell_max+1):
	N_TT[el-2] = ((D_T/T_CMB)**2)*np.exp(el*(el+1.0)*(fwmh**2)/8.0/np.log(2.0))

TT_unlensed = unlensed_spectra[:,1]/unlensed_spectra[:,0]/(unlensed_spectra[:,0]+1)*2.0*np.pi
TT_lensed = lens_potential_spectra[:,1]/lens_potential_spectra[:,0]/(lens_potential_spectra[:,0]+1)*2.0*np.pi

"""
np.savetxt('try1.dat', np.c_[ell,TT_lensed], fmt='%1.4e')
np.savetxt('try2.dat', np.c_[ell,N_TT], fmt='%1.4e')
"""

#signal + noise
#TT_unlensed = TT_unlensed + N_TT
TT_lensed = TT_lensed + N_TT

"""
np.savetxt('try3.dat', np.c_[ell,TT_lensed], fmt='%1.4e')
quit()
"""

aL_min = 2
aL_max = ell_lens_max
Aa = np.zeros(aL_max-aL_min+1)
L_array = np.zeros(aL_max-aL_min+1)
dtheta = 0.02
dl1 = 1

for aL in range(aL_min,aL_max+1,10): #list of Aa[aL]
    L_array[aL-2] = aL
    for theta in np.arange(0,2*np.pi,dtheta): #\theta integral
        for l1 in range(2,integral_ell_max): #ell_1 integral
            l2 = int(np.sqrt(l1**2+aL**2-2*aL*l1*np.cos(theta))) #value of l2 from triangle relation
            s = (aL+l1+l2)/2.0
            #if( (max(aL,l1,l2)<s) & (l2 < integral_ell_max) ): #triangle inequality & l2 < integral_ell_max
            #if(max(aL,l1,l2)<s): #triangle inequality
            #if(l2 < integral_ell_max):
            if(1<2):
                tmp = TT_unlensed[l1-2]*(aL*l1*np.cos(theta)) + TT_unlensed[l2-2]*((aL**2)-aL*l1*np.cos(theta))
                tmp = tmp**2
                tmp = tmp/2.0/TT_lensed[l1-2]/TT_lensed[l2-2]
                tmp = tmp * l1 * dtheta * dl1/4/np.pi/np.pi
                Aa[aL-2] = Aa[aL-2] + tmp
    Aa[aL-2] = aL*aL/Aa[aL-2]
    print aL,Aa[aL-2]

np.savetxt('noise.dat', np.c_[L_array,Aa], fmt='%1.4e')



