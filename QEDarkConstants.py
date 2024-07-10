## define constants
import numpy as np
sec2yr = 60.*60.*24*365.25 # sec in a year
sec2day = 1/(60.*60.*24)                                    # 1 s in days

lightSpeed = 299792458 # m / s
lightSpeed_kmpers = lightSpeed *1e-3
ccms = lightSpeed*1e2
hbar    = 6.582119569e-16 # eV * s
# alpha =  1.0/137.03599908 #fine structure constant
alpha = 1/137 #fine structure constant
me_eV = .511e6 # mass of electron in eV

# me_eV = 5.1099894e5 # mass of electron in eV
mP_eV = 938.27208816 *1e6 #mass of proton in eV
evtoj = 1.60218e-19 # J / eV
amu_nu = 9.314e8 # eV 
amu2kg = 1.660538782e-27 # amu to kg
pi = np.pi
cm2sec = 1/lightSpeed*1e-5

"""Thomas-Fermi Screening Parameters"""
eps0 = 11.3                                                 # Unitless
qTF = 4.13e3                                                # In eV
omegaP = 16.6                                               # In eV
alphaS = 1.563                                              # Unitless


"""Halo Model Parameters"""
q_Tsallis = 0.773
# v0_Tsallis = 267.2 #km/s
# vEsc_Tsallis = 560.8 #km/s
k_DPL = 2.0 #1.5 <= k <= 3.5 found to give best fit to N-body simulations. 
# p_MSW =  ?


"""Dark Matter Parameters"""
v0 = 238.0                                       # In units of km/s
vEarth = 250.2                                   # In units of km/s
vEscape = 544.0                                  # In units of km/s
rhoX = 0.3e9                                                # In eV/cm^3
crosssection = 1e-36                                        # In cm^2

## import QEdark data
nq = 900
nE = 500
dQ = .02*alpha*me_eV #eV
dE = 0.1 # eV
Emin = 0
Emax = 500*0.1
wk = 2/137
fcrys = {'Si': np.transpose(np.resize(np.loadtxt('./Si_f2.txt',skiprows=1),(nE,nq))),
         'Ge': np.transpose(np.resize(np.loadtxt('./Ge_f2.txt',skiprows=1),(nE,nq)))}

"""
    materials = {name: [Mcell #eV, Eprefactor, Egap #eV, epsilon #eV, fcrys]}
    N.B. If you generate your own fcrys from QEdark, please remove the factor of "wk/4" below. 
"""
materials = {'Si': [2*28.0855*amu2kg, 2.0, 1.2, 3.8,wk/4*fcrys['Si']], \
             'Ge': [2*72.64*amu2kg, 1.8, 0.7, 3,wk/4*fcrys['Ge']]} #could be 0.67, 2.8. not sure

