#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
2018/Oct BH @ BNU

Hu-Okamoto quatratic estimator
'''

import numpy as np
from scipy.integrate import dblquad,quad
from scipy import interpolate
from joblib import Parallel, delayed
import multiprocessing

ell_lens_min = 2
ell_lens_max = 1000
integral_ell_min = 2
integral_ell_max = 1500
tol = 1.0e-9

#noise params
#arcmin_to_radian
arcmin_to_radian = np.pi/180.0/60.0
#[\mu K]
T_CMB = 2.728E6
#Planck noise [\mu K*arcmin]
#D_T_arcmin = 35.4 #27.0
#D_P_arcmin = 63.1 #40.0*np.sqrt(2)
#Almost perfect noise [\mu K*arcmin]
#D_T_arcmin = 1.0
#D_P_arcmin = 1.0*np.sqrt(2)
#AliCPT noise [\mu K*arcmin]
D_T_arcmin = 9.0
D_P_arcmin = 9.0*np.sqrt(2)

#noise [\mu K*rad]
D_T = D_T_arcmin / arcmin_to_radian
D_P = D_P_arcmin / arcmin_to_radian

#Planck FWHM of the beam [arcmin]
#fwmh_arcmin = 7.0
#Almost perfect FWHM of the beam [arcmin]
#fwmh_arcmin = 4.0
#AliCPT FWHM of the beam [arcmin]
fwmh_arcmin = 12.0

fwmh = fwmh_arcmin * arcmin_to_radian


#file name
unlensed_spectra_filename = './data/lcdm_totCls.dat'
lensed_spectra_filename = './data/lcdm_lensedtotCls.dat'
lens_potential_filename = './data/lcdm_lenspotentialCls.dat'

#output_filename = 'new_planck_tt_noise2.dat'
#output_filename = 'new_planck_noise2.dat'
output_filename = 'alicpt_noise.dat'


#unlensed spectra
#un_Cl is ranged in: L, TT, EE, BB, TE
un_Cl = np.loadtxt(unlensed_spectra_filename)
#lensed spectra
#le_Cl is ranged in: L, TT, EE, BB, TE
le_Cl = np.loadtxt(lensed_spectra_filename)
#lensing potential spectra
lens_potential_spectra = np.loadtxt(lens_potential_filename)

#number of ell_max
l_un_max = len(un_Cl[:,0])+1
l_le_max = len(le_Cl[:,0])+1

'''
np.savetxt('try1.dat', un_Cl, fmt='%1.4e')
np.savetxt('try2.dat', le_Cl, fmt='%1.4e')
'''

#noise spectrum (only added to the lensed spectra)
N_TT = np.zeros(l_le_max-2+1)
N_EE = np.zeros(l_le_max-2+1)
N_BB = np.zeros(l_le_max-2+1)
for el in range(2,l_le_max+1):
    N_TT[el-2] = ((D_T/T_CMB)**2)*np.exp(el*(el+1.0)*(fwmh**2)/8.0/np.log(2.0))
    N_EE[el-2] = ((D_P/T_CMB)**2)*np.exp(el*(el+1.0)*(fwmh**2)/8.0/np.log(2.0))
    N_BB[el-2] = ((D_P/T_CMB)**2)*np.exp(el*(el+1.0)*(fwmh**2)/8.0/np.log(2.0))

#remove the ell scaling
un_Cl[:,1] = un_Cl[:,1]/un_Cl[:,0]/(un_Cl[:,0]+1)*2.0*np.pi #TT
un_Cl[:,2] = un_Cl[:,2]/un_Cl[:,0]/(un_Cl[:,0]+1)*2.0*np.pi #EE
un_Cl[:,3] = un_Cl[:,3]/un_Cl[:,0]/(un_Cl[:,0]+1)*2.0*np.pi #BB
un_Cl[:,4] = un_Cl[:,4]/un_Cl[:,0]/(un_Cl[:,0]+1)*2.0*np.pi #TE

le_Cl[:,1] = le_Cl[:,1]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi #TT
le_Cl[:,2] = le_Cl[:,2]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi #EE
le_Cl[:,3] = le_Cl[:,3]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi #BB
le_Cl[:,4] = le_Cl[:,4]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi #TE

'''
np.savetxt('try3.dat', un_Cl, fmt='%1.4e')
np.savetxt('try4.dat', le_Cl, fmt='%1.4e')
np.savetxt('try5.dat', np.c_[le_Cl[:,0],N_TT,N_EE], fmt='%1.4e')
'''

#lensed signal + noise
le_Cl[:,1] = le_Cl[:,1] + N_TT
le_Cl[:,2] = le_Cl[:,2] + N_EE
le_Cl[:,3] = le_Cl[:,3] + N_BB

#np.savetxt('try6.dat', le_Cl, fmt='%1.4e')

#Perform the interpolation, the interpolation range may outof bound of sampling points
#interpolate the unlensed tt, if out of the sampling range, we set to zero
#noise has been added
Cl_tt = interpolate.interp1d(un_Cl[:,0],un_Cl[:,1],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed ee
Cl_ee = interpolate.interp1d(un_Cl[:,0],un_Cl[:,2],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed bb
Cl_bb = interpolate.interp1d(un_Cl[:,0],un_Cl[:,3],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed te
Cl_te = interpolate.interp1d(un_Cl[:,0],un_Cl[:,4],bounds_error=False,kind='linear',fill_value=tol)

#Perform the interpolation, the interpolation range may outof bound of sampling points
#interpolate the lensed tt, if out of the sampling range, we set to zero
lCl_tt = interpolate.interp1d(le_Cl[:,0],le_Cl[:,1],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed ee
lCl_ee = interpolate.interp1d(le_Cl[:,0],le_Cl[:,2],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed bb
lCl_bb = interpolate.interp1d(le_Cl[:,0],le_Cl[:,3],bounds_error=False,kind='linear',fill_value=tol)
#interpolate the unlensed te
lCl_te = interpolate.interp1d(le_Cl[:,0],le_Cl[:,4],bounds_error=False,kind='linear',fill_value=tol)

#define f_a function
def fa(ea,el1,eL,ecos_phi1,esgn_phi12):
    '''
    ea: (int) index of estimator
    el1: (int) l1 module
    eL: (int) L module
    ecos_phi1: (float) phi1 is the azimuthal angle btw l1 and L,hence phi1 \in (0,2pi)
    esgn_phi12: sign of phi_12
    '''
    
    el2 = int(np.sqrt(eL**2+el1**2-2*el1*eL*ecos_phi1)) #value of l2 from triangle relation
    gl2 = np.sqrt(eL**2+el1**2-2*el1*eL*ecos_phi1) #float l2
    if((el2==0) or (el2==1)):
        fa = 0.0 #avoid these multiples
        return fa
    L_dot_l1 = eL*el1*ecos_phi1
    L_dot_l2 = eL**2-L_dot_l1
    
    if(abs(ecos_phi1)>1):
        print('fa: cos_phi1 is larger than unity!')
        quit()
    
    if(esgn_phi12<0):
        esin_phi1 = np.sqrt(1.0-ecos_phi1**2)
    else:
        esin_phi1 = -np.sqrt(1.0-ecos_phi1**2)

    ecos_phi2 = (gl2**2+eL**2-el1**2)/2./gl2/eL
    esin_phi2 = np.sqrt(1.-ecos_phi2**2)

    if(esgn_phi12<0):
        ecos_phi12 = ecos_phi1*ecos_phi2 - esin_phi1*esin_phi2
        esin_phi12 = -np.sqrt(1-ecos_phi12**2)
    else:
        ecos_phi12 = ecos_phi1*ecos_phi2 + esin_phi1*esin_phi2
        esin_phi12 = np.sqrt(1-ecos_phi12**2)

    ecos_2phi = 2.*ecos_phi12**2-1.0
    esin_2phi = 2.*esin_phi12*ecos_phi12

    if(ea == 1):   #TT
        fa = Cl_tt(el1)*(L_dot_l1)+Cl_tt(el2)*(L_dot_l2)
    elif(ea == 2): #TE
        fa = Cl_te(el1)*ecos_2phi*(L_dot_l1)+Cl_te(el2)*(L_dot_l2)
    elif(ea == 3): #TB
        fa = Cl_te(el1)*esin_2phi*(L_dot_l1)
    elif(ea == 4): #EE
        fa = (Cl_ee(el1)*(L_dot_l1)+Cl_ee(el2)*(L_dot_l2))*ecos_2phi
    elif(ea == 5): #EB
        fa = (Cl_ee(el1)*(L_dot_l1)-Cl_bb(el2)*(L_dot_l2))*esin_2phi
    else:
        print('f_a: check index a value! have to be in the range of (1,5)')
        quit()

    return fa

#define F_a function
def cf(xa,xl1,xL,xcos_phi1,xsgn_phi12):
    '''
    xa: (int) index of estimator
    xl1: (int) l1 module
    xL: (int) L module
    xcos_phi1: (float) phi1 is the azimuthal angle btw l1 and L,hence phi1 \in (0,2pi)
    xsgn_phi12: sign of phi_12
    '''

    xsgn_phi21 = -xsgn_phi12
    xl2 = int(np.sqrt(xL**2+xl1**2-2*xl1*xL*xcos_phi1)) #value of l2 from triangle relation
    fl2 = np.sqrt(xL**2+xl1**2-2*xl1*xL*xcos_phi1) #float l2
    if((xl2==0) or (xl2==1)):
        cf = 0.0 #avoid these multiples
        return cf
    xcos_phi2 = (fl2**2+xL**2-xl1**2)/2./fl2/xL
    
    if(xa == 1):   #TT
        cf = fa(xa,xl1,xL,xcos_phi1,xsgn_phi12)/2.0/lCl_tt(xl1)/lCl_tt(xl2)
    elif(xa == 2): #TE
        cf = fa(xa,xl1,xL,xcos_phi1,xsgn_phi12)*lCl_ee(xl1)*lCl_tt(xl2) - fa(xa,xl2,xL,xcos_phi2,xsgn_phi21)*lCl_te(xl1)*lCl_te(xl2)
        tmp2 = lCl_tt(xl1)*lCl_ee(xl2)*lCl_ee(xl1)*lCl_tt(xl2)
        tmp2 = tmp2 - (lCl_te(xl1)*lCl_te(xl2))**2
        cf = cf/tmp2
    elif(xa == 3): #TB
        cf = fa(xa,xl1,xL,xcos_phi1,xsgn_phi12)/lCl_tt(xl1)/lCl_bb(xl2)
    elif(xa == 4): #EE
        cf = fa(xa,xl1,xL,xcos_phi1,xsgn_phi12)/2.0/lCl_ee(xl1)/lCl_ee(xl2)
    elif(xa == 5): #EB
        cf = fa(xa,xl1,xL,xcos_phi1,xsgn_phi12)/lCl_ee(xl1)/lCl_bb(xl2)
    else:
        print('F_a: check index a value! have to be in the range of (1,5)')
        quit()

    return cf

#define A_a function
def Aa(L,a):
    '''
    L: (int) L module
    a: (int) index of estimator
    '''
    dphi1 = 0.02
    dl1 = 1
    Aa = 0.0
    for l1 in range(integral_ell_min,integral_ell_max): #ell_1 integral
        for phi1 in np.arange(0,2*np.pi,dphi1): #phi1 integral
            
            l2 = int(np.sqrt(L**2+l1**2-2*l1*L*np.cos(phi1))) #value of l2 from triangle relation
            
            if((l2==0) or (l2==1)):
                tmp = 0.0 #avoid these multiples
            else:
                if((phi1>=0) and (phi1<np.pi)):
                    sgn_phi12 = -1
                else:
                    sgn_phi12 = 1
                cos_phi1 = np.cos(phi1)
                tmp = fa(a,l1,L,cos_phi1,sgn_phi12)*cf(a,l1,L,cos_phi1,sgn_phi12)
                tmp = tmp * l1 * dphi1 * dl1/4/np.pi/np.pi

            Aa = Aa + tmp

    Aa = L*L/Aa
    return Aa

#N_aa=Aa term, for test
L_min = ell_lens_min
L_max = ell_lens_max
L_array = np.zeros(L_max-L_min+1)
Ntt_array = np.zeros(L_max-L_min+1)
Nte_array = np.zeros(L_max-L_min+1)
Ntb_array = np.zeros(L_max-L_min+1)
Nee_array = np.zeros(L_max-L_min+1)
Neb_array = np.zeros(L_max-L_min+1)
op_row = np.zeros(6)
l_range = range(L_min,L_max+1,1)

def NoiseOutPut(yL):
    op_data = open(output_filename,'a')
    print('L=',yL)
    L_array[yL-L_min] = yL
    Ntt_array[yL-L_min] = Aa(yL,1) #N_tt
    Nte_array[yL-L_min] = Aa(yL,2) #N_te
    Ntb_array[yL-L_min] = Aa(yL,3) #N_tb
    Nee_array[yL-L_min] = Aa(yL,4) #N_ee
    Neb_array[yL-L_min] = Aa(yL,5) #N_eb
    op_row[0] = L_array[yL-L_min]
    op_row[1] = Ntt_array[yL-L_min]
    op_row[2] = Nte_array[yL-L_min]
    op_row[3] = Ntb_array[yL-L_min]
    op_row[4] = Nee_array[yL-L_min]
    op_row[5] = Neb_array[yL-L_min]
    np.savetxt(op_data, op_row.reshape(1, op_row.shape[0]), fmt='%1.4e')
    op_data.close()
    return None

#parallel the Noise output part
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(NoiseOutPut)(zL) for zL in l_range)

exit()
