import numpy as np
import scipy.constants as ct
import scipy.integrate as integrate
from astropy.cosmology import LambdaCDM
from astropy.cosmology import WMAP9
from numpy.linalg import inv
import math as mh
import pdb


G= ct.G
c= ct.c
pi = ct.pi
Msun = 1.989e30

"""
inspiral_SNR_calc numerically calculates the SNR of a given binary black hole realized during the inspiral phase of merger. This is based on Flanagan and Hughes (1998) with some small additions. 


inspiral_SNR_calc takes the following arguments:

total_mass (Array-like) --> Solar Masses, LISA refernce frame 
mass_ratio (Array-like)
redshift   (Array-like)
start_time/end_time (Array-like or float, must be same length) --> Years before merger
frq (1D array) --> Hertz,  frequency value of each noise curve point. 
hf (1D array) --> per root Hertz,  one-sided power spectral density of the noise in the detector
 
"""
def inspiral_SNR_calc(total_mass,mass_ratio, z, start_time, end_time, frq, hf):
	#convert values to geometricized units
	frq = frq/c
	hf = hf*c**(1.0/2.0)
	total_mass = total_mass*G*Msun/(c*c)
	start_time = start_time*365.25*24.0*3600.0*c
	end_time = end_time*365.25*24.0*3600.0*c

	reduced_mass = np.multiply(total_mass,mass_ratio)/(1.0+mass_ratio)**2.0
	mrgfrq = 0.02/total_mass

	cosmo = LambdaCDM(WMAP9.H(0.0).value, WMAP9.Om(0.0), 1.0-WMAP9.Om(0.0))
	D_L = (cosmo.luminosity_distance(z).value)*1e6*3.086e16


	fstart = ((mrgfrq**(-8.0/3.0))+256.0/5.0*(pi**(8.0/3.0))*reduced_mass*(total_mass**(2.0/3.0))*start_time)**(-3.0/8.0)
	fend = ((mrgfrq**(-8.0/3.0))+256.0/5.0*(pi**(8.0/3.0))*reduced_mass*(total_mass**(2.0/3.0))*end_time)**(-3.0/8.0)
	
	prefactor = prefactorfunc_ins(total_mass, reduced_mass, D_L)
	integral = integralfunc_ins(fstart, fend, frq, hf)
	SNR = (np.multiply(prefactor,integral))**0.5


	#error calculations

	hfmin = np.argmin(hf)
	f0 = frq[hfmin]
		
	f1 = frq[0]
	f2 = frq[len(frq)-1]
	alpha = np.array([-1.0/3.0, -7.0/3.0, -17.0/3.0, -4.0/3.0, -4.0,-3.0])
	I = np.zeros(len(alpha))
	i = 0
	for a in alpha:
		trans, err = integrate.quad(Ialphafunc, f1, f2, args = (a, frq, hf), epsabs = 0, limit=1000)
		I[i] = trans
		i+=1
	
	A1 = np.array([f0**(-(alpha[0]-alpha[1]))*(I[0]/I[1]), f0**(-(alpha[3]-alpha[1]))*(I[3]/I[1]),f0**(-(alpha[5]-alpha[1]))*(I[5]/I[1])])
	A2 = np.array([f0**(-(alpha[3]-alpha[1]))*(I[3]/I[1]), f0**(-(alpha[1]-alpha[1]))*(I[1]/I[1]),f0**(-(alpha[4]-alpha[1]))*(I[4]/I[1])])
	A3 = np.array([f0**(-(alpha[5]-alpha[1]))*(I[5]/I[1]), f0**(-(alpha[4]-alpha[1]))*(I[4]/I[1]),f0**(-(alpha[2]-alpha[1]))*(I[2]/I[1])])

	Aij = np.array([A1,A2, A3])
	
	Bij = inv(Aij)

	sigmaD = 1.0/SNR
	sigmat0 = 1.0/SNR*mh.fabs(Bij[0][0])/(2.0*pi*f0)/start_time
	sigmaphi = 1.0/SNR*mh.fabs(Bij[1][1])/2.0
	sigmachirpM = 1.0/SNR*(128.0/5.0)*(pi*(reduced_mass*total_mass**(2.0/3.0))**(3.0/5.0)*f0)**(5.0/3.0)*mh.fabs(Bij[2][2])
							
	outdict = {'SNR':SNR, 'sigma_D':sigmaD, 'sigma_t0':sigmat0,'sigma_phi':sigmaphi,'sigma_chirpM':sigmachirpM} 
	return outdict



"""
ringdown_SNR_calc numerically calculates the SNR of a given binary black hole realized during the ringdown phase of coalescence. This is based on Flanagan and Hughes (1998) with some small additions. 


ringdown_SNR_calc takes the following arguments:

total_mass (Array-like) --> Solar Masses 
mass_ratio (Array-like)
redshift   (Array-like)
spin	   (Array-like)--> dimensionless [0,0.98]
frq (1D array) --> Hertz,  frequency value of each noise curve point. 
hf (1D array) --> per root Hertz,  one-sided power spectral density of the noise in the detector
 
"""
def ringdown_SNR_calc(total_mass,mass_ratio, z, a, frq, hf):
	# conrolling mode of ringdown
	f1 = 1.5251
	f2 = -1.1568
	f3 = 0.1292
	q1 = 0.7
	q2 = 1.4187
	q3 = -0.499
	
	#convert values to geometricized units
	frq = frq/c
	hf = hf*c**(1.0/2.0)
	total_mass = total_mass*G*Msun/(c*c)


	fqnr = (f1+f2*(1.0-a)**f3)*1.0/(2.0*pi*total_mass)
	Q = (q1+q2*(1.0-a)**q3)
	decaytime = Q/(pi*fqnr)
	Qprime = -q2*q3*(1.0-a)**(q3-1.0)
	fprime =-f2*f3*(1.0-a)**(f3-1.0)*1.0/(2.0*pi*total_mass)


	fstart = np.zeros(len(fqnr))
	fend = fqnr*1.0e3

	reduced_mass = np.multiply(total_mass,mass_ratio)/(1.0+mass_ratio)**2.0

	Amplitude=(0.03*128.0*reduced_mass*reduced_mass/(total_mass*total_mass*total_mass)*1.0/(fqnr*Q))**(1.0/2.0)

	cosmo = LambdaCDM(WMAP9.H(0.0).value, WMAP9.Om(0.0), 1.0-WMAP9.Om(0.0))
	D_L = (cosmo.luminosity_distance(z).value)*1e6*3.086e16
	
	prefactor = prefactorfunc_rd(total_mass, reduced_mass, D_L, Amplitude, decaytime)
	integral = integralfunc_rd(fstart, fend,fqnr, decaytime, frq, hf)
	SNR = (np.multiply(prefactor,integral))**0.5
	

	#Error Calculations

	sigma_J= (1.0/SNR)*np.fabs(2.0*Q/Qprime)
	sigma_M= (1.0/SNR)*np.fabs(2.0*Q*fprime/(fqnr*Qprime))
	sigma_A= (1.0/SNR)*Amplitude*2.0**(1.0/2.0)
	sigma_phi= (1.0/SNR)
	sigma_fqnr= (1.0/SNR)*1.0/(2.0**(1.0/2.0)*pi*decaytime)*1.0/fqnr
	sigma_dt= (2.0/SNR)

	outdict = {'SNR':SNR, 'sigma_J':sigma_J, 'sigma_M':sigma_M,'sigma_phi':sigma_phi,'sigma_fqnr':sigma_fqnr,'sigma_dt':sigma_dt} 	
	return outdict


"""
inspiral_local_SNR_calc numerically calculates the SNR of a given binary black hole realized during the inspiral phase of merger. This is based on Flanagan and Hughes (1998) with some small additions. 


inspiral_local_SNR_calc takes the following arguments:

total_mass (Array-like) --> Solar Masses, in LISA reference frame
mass_ratio (Array-like)
Distance   (Array-like) --> pc
start_time/end_time (Array-like or float, must be same length) --> Years before merger
frq (1D array) --> Hertz,  frequency value of each noise curve point. 
hf (1D array) --> per root Hertz,  one-sided power spectral density of the noise in the detector
 
"""
def inspiral_local_SNR_calc(total_mass,mass_ratio, D, start_time, end_time, frq, hf):
	#convert values to geometricized units
	frq = frq/c
	hf = hf*c**(1.0/2.0)
	total_mass = total_mass*G*Msun/(c*c)
	start_time = start_time*365.25*24.0*3600.0*c
	end_time = end_time*365.25*24.0*3600.0*c
	D = D*3.086e16

	reduced_mass = np.multiply(total_mass,mass_ratio)/(1.0+mass_ratio)**2.0
	mrgfrq = 0.02/total_mass



	fstart = ((mrgfrq**(-8.0/3.0))+256.0/5.0*(pi**(8.0/3.0))*reduced_mass*(total_mass**(2.0/3.0))*start_time)**(-3.0/8.0)
	fend = ((mrgfrq**(-8.0/3.0))+256.0/5.0*(pi**(8.0/3.0))*reduced_mass*(total_mass**(2.0/3.0))*end_time)**(-3.0/8.0)
	
	prefactor = prefactorfunc_ins(total_mass, reduced_mass, D)
	integral = integralfunc_ins(fstart, fend, frq, hf)
	SNR = (np.multiply(prefactor,integral))**0.5


	#error calculations

	hfmin = np.argmin(hf)
	f0 = frq[hfmin]
		
	f1 = frq[0]
	f2 = frq[-1]
	alpha = np.array([-1.0/3.0, -7.0/3.0, -17.0/3.0, -4.0/3.0, -4.0,-3.0])
	I = np.zeros(len(alpha))
	i = 0
	for a in alpha:
		trans, err = integrate.quad(Ialphafunc, f1, f2, args = (a, frq, hf), epsabs = 0, limit=1000)
		I[i] = trans
		i+=1
	
	A1 = np.array([f0**(-(alpha[0]-alpha[1]))*(I[0]/I[1]), f0**(-(alpha[3]-alpha[1]))*(I[3]/I[1]),f0**(-(alpha[5]-alpha[1]))*(I[5]/I[1])])
	A2 = np.array([f0**(-(alpha[3]-alpha[1]))*(I[3]/I[1]), f0**(-(alpha[1]-alpha[1]))*(I[1]/I[1]),f0**(-(alpha[4]-alpha[1]))*(I[4]/I[1])])
	A3 = np.array([f0**(-(alpha[5]-alpha[1]))*(I[5]/I[1]), f0**(-(alpha[4]-alpha[1]))*(I[4]/I[1]),f0**(-(alpha[2]-alpha[1]))*(I[2]/I[1])])

	Aij = np.array([A1,A2, A3])
	
	Bij = inv(Aij)

	sigmaD = 1.0/SNR
	sigmat0 = 1.0/SNR*mh.fabs(Bij[0][0])/(2.0*pi*f0)/start_time
	sigmaphi = 1.0/SNR*mh.fabs(Bij[1][1])/2.0
	sigmachirpM = 1.0/SNR*(128.0/5.0)*(pi*(reduced_mass*total_mass**(2.0/3.0))**(3.0/5.0)*f0)**(5.0/3.0)*mh.fabs(Bij[2][2])
							
	outdict = {'SNR':SNR, 'sigma_D':sigmaD, 'sigma_t0':sigmat0,'sigma_phi':sigmaphi,'sigma_chirpM':sigmachirpM} 
	return outdict






"""
Supporting Functions:
"""

def prefactorfunc_ins(total_mass, reduced_mass, lumdis):
	return 2.0/(3.0*5.0)*total_mass**(2.0/3.0)*reduced_mass*pi**(-4.0/3.0)*lumdis**(-2.0)

def prefactorfunc_rd(total_mass, reduced_mass, lumdis, Amplitude, decaytime):
	return (2.0)/(32.0*5.0)*Amplitude**2.0*total_mass**2.0*1.0/(pi**5.0*lumdis**2.0*decaytime**2.0)


def integralfunc_ins(fstart, fend, frq, hf):
	out = np.zeros(len(fstart))
	i = 0
	for f1 in fstart:
		transfer, err = integrate.quad(NoiseCurveint_ins, f1, fend[i], args = (frq, hf), epsabs = 0, limit=1000) #must add or remove the averaging factor of 5 to the numerator based on using Sh or hn. 
		out[i] = transfer
			
		i+=1
	return out

def NoiseCurveint_ins(f_m, frq, hf):
	if f_m == 0.0:
		return 0.0
	log10f_m = np.log10(f_m)
	frqlog10 = np.log10(frq)
	hflog10 = np.log10(hf)
	hfnoiselog10 = np.interp(log10f_m, frqlog10, hflog10, left = frq[0], right = frq[-1])
	hfnoise = 10.0**hfnoiselog10
	return 1.0/(f_m**(7.0/3.0)*hfnoise**2.0)


def integralfunc_rd(fstart, fend, fqnr, decaytime, frq, hf):
	out = np.zeros(len(fqnr))
	i = 0
	for f1 in fstart:
		transfer, err = integrate.quad(NoiseCurveint_rd, f1, fend[i], args = (fqnr[i], decaytime[i], frq, hf), epsabs = 0, limit=1000) #must add or remove the averaging factor of 5 to the numerator based on using Sh or hn. 
		out[i] = transfer
			
		i+=1
	return out

def NoiseCurveint_rd(f_m, fqnr, decaytime, frq, hf):
	if f_m == 0.0:
		return 0.0
	log10f_m = np.log10(f_m)
	frqlog10 = np.log10(frq)
	hflog10 = np.log10(hf)
	hfnoiselog10 = np.interp(log10f_m, frqlog10, hflog10, left = frq[0], right = frq[-1])
	hfnoise = 10.0**hfnoiselog10
	return 1.0/hfnoise**2.0*(1.0/((f_m-fqnr)**2.0+(2*pi*decaytime)**(-2.0))**2.0+1.0/((f_m+fqnr)**2.0+(2*pi*decaytime)**(-2.0))**2.0)



def Ialphafunc(f_m, alpha, frq, hf):
	if f_m == 0.0:
		return 0.0
	log10f_m = np.log10(f_m)
	frqlog10 = np.log10(frq)
	hflog10 = np.log10(hf)
	hfnoiselog10 = np.interp(log10f_m, frqlog10, hflog10, left = frq[0], right = frq[-1])
	hfnoise = 10.0**hfnoiselog10
	integrand = f_m**alpha/hfnoise
	return integrand


#D = np.logspace(1.0, 7.0, 5)
#total_mass = np.logspace(-2.0,1.5,5)
#mass_ratio = np.linspace(0.1, 1.0, 5)
#a = np.linspace(0.6, 0.98, 5)
#start_time = 1.0
#end_time = 0.0
#frq = np.logspace(-8.0, 1.0, 1000)
#hf = np.full(1000,1e-20)


#out = inspiral_local_SNR_calc(total_mass, mass_ratio, D, start_time, end_time, frq, hf)
#pdb.set_trace()


