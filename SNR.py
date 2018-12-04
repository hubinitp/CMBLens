#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
2018/Nov BH @ BNU
'''

import numpy as np

#noise spectra name
#noise_spectra_filename = './alicpt_noise2.dat'
#noise_spectra_filename = './planck_noise_mv.dat'
noise_spectra_filename = './ideal_noise_mv.dat'
#signal spectra name
signal_spectra_filename = './data/lcdm_lenspotentialCls.dat'


#noise spectra
#spec is ranged in: L, TT_noise, TE_noise, TB_noise, EE_noise, EB_noise, mv_noise
Noise = np.loadtxt(noise_spectra_filename)

#signal spectra
#dd spectrum is sorted in the sixth column with ell*(ell+1)/2/pi
Signal = np.loadtxt(signal_spectra_filename)

#rm the prefactors
Signal[:,5] = Signal[:,5]/Signal[:,0]/(Signal[:,0]+1)*2.0*np.pi #TT

#number of ell_max
l_max = len(Noise[:,0])+1

SNR = 0.0

for l in range(2,l_max+1):
    SNR = SNR + Signal[l-2,5]/Noise[l-2,6]

np.savetxt('check.dat', np.c_[Signal[0:l_max-1,0],Signal[0:l_max-1,5],Noise[0:l_max-1,0],Noise[0:l_max-1,6]], fmt='%1.4e')

print('SNR_mv = ',np.sqrt(SNR))

exit()
