#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
2018/Oct BH @ BNU

Hu-Okamoto quatratic estimator
'''

import numpy as np
from scipy.integrate import dblquad,quad
from scipy import interpolate

ell_lens_max = 1000
integral_ell_max = 3000
tol = 1.0e-9

#noise params
#arcmin_to_radian
arcmin_to_radian = np.pi/180.0/60.0
#[\mu K]
T_CMB = 2.728E6
#Planck noise [\mu K*arcmin]
D_T_arcmin = 27.0
D_P_arcmin = 40.0*np.sqrt(2)
#Almost perfect noise [\mu K*arcmin]
#D_T_arcmin = 1.0
#D_P_arcmin = 1.0*np.sqrt(2)

#noise [\mu K*rad]
D_T = D_T_arcmin / arcmin_to_radian
D_P = D_P_arcmin / arcmin_to_radian

#Planck FWHM of the beam [arcmin]
fwmh_arcmin = 7.0
#Almost perfect FWHM of the beam [arcmin]
#fwmh_arcmin = 4.0

fwmh = fwmh_arcmin * arcmin_to_radian


#file name
unlensed_spectra_filename = './data/lcdm_totCls.dat'
lensed_spectra_filename = './data/lcdm_lensedtotCls.dat'
lens_potential_filename = './data/lcdm_lenspotentialCls.dat'

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

le_Cl[:,1] = le_Cl[:,1]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi
le_Cl[:,2] = le_Cl[:,2]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi
le_Cl[:,3] = le_Cl[:,3]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi
le_Cl[:,4] = le_Cl[:,4]/le_Cl[:,0]/(le_Cl[:,0]+1)*2.0*np.pi

'''
np.savetxt('try3.dat', un_Cl, fmt='%1.4e')
np.savetxt('try4.dat', le_Cl, fmt='%1.4e')
np.savetxt('try5.dat', np.c_[le_Cl[:,0],N_TT,N_EE], fmt='%1.4e')
'''

#lensed signal + noise
le_Cl[:,1] = le_Cl[:,1] + N_TT
le_Cl[:,2] = le_Cl[:,2] + N_EE
le_Cl[:,3] = le_Cl[:,3] + N_BB

#define f_a function
def fa(a,l1,l2,cos_theta,sin_phi12):
    '''
    a: (int) index of estimator
    l1: (int) l1 module
    l2: (int) l2 module
    cos_theta: (float) theta is the azimuthal angle btw l1 and l2,hence theta \in (0,pi)
    sin_phi12: (float) an indicator of phi_12
    '''
    
    L = int(np.sqrt(l1**2+l2**2+2*l1*l2*cos_theta)) #value of L from triangle relation
    L_dot_l1 = l1**2+l1*l2*cos_theta
    L_dot_l2 = l2**2+l1*l2*cos_theta
    cos_phi = cos_theta
    if(abs(cos_phi)>1):
        print('fa: cos_phi is larger than unity!')
        quit()
    
    if((sin_phi12<=0) and (sin_phi12>=-1)):
        sin_phi = -np.sqrt(1.0-cos_phi**2)
    elif((sin_phi12>0) and (sin_phi12<=1)):
        sin_phi = np.sqrt(1.0-cos_phi**2)
    else:
        print('f_a: sin_phi12 have to be in the range of (-1,1)')
        quit()

    cos_2phi = 2*sin_phi*cos_phi
    sin_2phi = 2*cos_phi**2-1.0

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

    if(a == 1):   #TT
        fa = Cl_tt(l1)*(L_dot_l1)+Cl_tt(l2)*(L_dot_l2)
    elif(a == 2): #TE
        fa = Cl_te(l1)*cos_2phi*(L_dot_l1)+Cl_te(l2)*(L_dot_l2)
    elif(a == 3): #TB
        fa = Cl_te(l1)*sin_2phi*(L_dot_l1)
    elif(a == 4): #EE
        fa = (Cl_ee(l1)*(L_dot_l1)+Cl_ee(l2)*(L_dot_l2))*cos_2phi
    elif(a == 5): #EB
        fa = (Cl_ee(l1)*(L_dot_l1)-Cl_bb(l2)*(L_dot_l2))*sin_2phi
    else:
        print('f_a: check index a value! have to be in the range of (1,5)')
        quit()

    return fa

#define F_a function
def cf(a,l1,l2,cos_theta,sin_phi12):
    '''
    a: (int) index of estimator
    l1: (int) l1 module
    l2: (int) l2 module
    cos_theta: (float) theta is the azimuthal angle btw l1 and l2,hence theta \in (0,pi)
    sin_phi12: (float) an indicator of phi_12
    '''
    
    #Perform the interpolation, the interpolation range may outof bound of sampling points
    #interpolate the lensed tt, if out of the sampling range, we set to zero
    lCl_tt = interpolate.interp1d(le_Cl[:,0],le_Cl[:,1],bounds_error=False,kind='linear',fill_value=tol)
    #interpolate the unlensed ee
    lCl_ee = interpolate.interp1d(le_Cl[:,0],le_Cl[:,2],bounds_error=False,kind='linear',fill_value=tol)
    #interpolate the unlensed bb
    lCl_bb = interpolate.interp1d(le_Cl[:,0],le_Cl[:,3],bounds_error=False,kind='linear',fill_value=tol)
    #interpolate the unlensed te
    lCl_te = interpolate.interp1d(le_Cl[:,0],le_Cl[:,4],bounds_error=False,kind='linear',fill_value=tol)

    if(a == 1):   #TT
        cf = fa(a,l1,l2,cos_theta,sin_phi12)/2.0/lCl_tt(l1)/lCl_tt(l2)
    elif(a == 2): #TE
        cf = fa(a,l1,l2,cos_theta,sin_phi12)*lCl_ee(l1)*lCl_tt(l2) - fa(a,l2,l1,cos_theta,-sin_phi12)*lCl_te(l1)*lCl_te(l2)
        tmp = lCl_tt(l1)*lCl_ee(l2)*lCl_ee(l1)*lCl_tt(l2)
        tmp = tmp - (lCl_te(l1)*lCl_te(l2))**2
        cf = cf/tmp
    elif(a == 3): #TB
        cf = fa(a,l1,l2,cos_theta,sin_phi12)/lCl_tt(l1)/lCl_bb(l2)
    elif(a == 4): #EE
        cf = fa(a,l1,l2,cos_theta,sin_phi12)/2.0/lCl_ee(l1)/lCl_ee(l2)
    elif(a == 5): #EB
        cf = fa(a,l1,l2,cos_theta,sin_phi12)/lCl_ee(l1)/lCl_bb(l2)
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
    for l1 in range(2,integral_ell_max): #ell_1 integral
        for phi1 in np.arange(0,2*np.pi,dphi1): #phi1 integral
            l2 = int(np.sqrt(l1**2+L**2-2*L*l1*np.cos(phi1))) #value of l2 from triangle relation
            
            if(l2==0 or l2==1): #avoid these multipoles
                l2 = 2 #brutal force
            
            '''
            if(l2 == 0):
                cos_tphi2 = 0.0
            else:
                cos_tphi2 = (l2**2+L**2-l1**2)/2/L/l2
            '''
            cos_tphi2 = (l2**2+L**2-l1**2)/2/L/l2
            
            if(abs(cos_tphi2)>1):
                print('Aa: cos_tphi2 is larger than unity!')
                quit()
            sin_tphi2 = np.sqrt(1-cos_tphi2**2)
            if((phi1>=0) and (phi1<=np.pi)):
                cos_theta = np.cos(phi1)*cos_tphi2-np.sin(phi1)*sin_tphi2
            else:
                cos_theta = np.cos(phi1)*cos_tphi2+np.sin(phi1)*sin_tphi2

            sin_phi12 = -np.sin(phi1)
            s = (L+l1+l2)/2.0
            if(max(L,l1,l2)<s): #triangle inequality
                tmp = fa(a,l1,l2,cos_theta,sin_phi12)*cf(a,l1,l2,cos_theta,sin_phi12)
                tmp = tmp * l1 * dphi1 * dl1/4/np.pi/np.pi
                Aa = Aa + tmp
    Aa = L*L/Aa
    return Aa

#N_aa=Aa term, for test
L_min = 10
L_max = ell_lens_max
L_array = np.zeros(L_max-L_min+1)
Ntt_array = np.zeros(L_max-L_min+1)
Nte_array = np.zeros(L_max-L_min+1)
Ntb_array = np.zeros(L_max-L_min+1)
Nee_array = np.zeros(L_max-L_min+1)
Neb_array = np.zeros(L_max-L_min+1)
for L in range(L_min,L_max+1,1): #list of Aa[L]
    print('L=',L)
    L_array[L-L_min] = L
    Ntt_array[L-L_min] = Aa(L,1) #N_tt
    Nte_array[L-L_min] = Aa(L,2) #N_te
    Ntb_array[L-L_min] = Aa(L,3) #N_tb
    Nee_array[L-L_min] = Aa(L,4) #N_ee
    Neb_array[L-L_min] = Aa(L,5) #N_eb


np.savetxt('new_planck_tt_noise.dat', np.c_[L_array,Ntt_array,Nte_array,Ntb_array,Nee_array,Neb_array], fmt='%1.4e')
    
