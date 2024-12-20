# -*- coding: utf-8 -*-
"""
This script provides common Elastic modulus parametrizations on density
which can also be scaled to measured data points

"""

import numpy as np
import uncertainties

#%% getting Elastic moduli from densities
#Gerling et al AC
def e_gerling_2017_AC(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    if scaling == None:
        print('no scaling used')
        return(6E-10*rho**4.6 * 1E6)
    else:
        try:
            factor = scaling[0] / (6E-10*scaling[1]**4.6)
            return(factor * 6E-10*rho**4.6* 1E6)
        except:
            print('scaling did not work')
            return(False)
e_gerling_2017_AC_with_unc = uncertainties.wrap(e_gerling_2017_AC)

 #Gerling et al CT
def e_gerling_2017_CT(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    try:
        factor = scaling[0] / 2E-8*scaling[1]**3.98
        return(factor * 2E-8*rho**3.98 * 1E6)
    except: 
        print('no scaling used')
        return(2E-8*rho**3.98 * 1E6)
e_gerling_2017_CT_with_unc = uncertainties.wrap(e_gerling_2017_CT)

 #Bergfeld et al 2022
def e_bergfeld_2022(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    try:
        factor = scaling[0] / 6.5E3*scaling[1]**4.4
        return(factor * 6.5E3*(rho/918)**4.4 * 1E6)
    except: 
        print('no scaling used')
        return(6.5E3*(rho/918)**4.4 * 1E6)
e_bergfeld_2022_with_unc = uncertainties.wrap(e_bergfeld_2022)
            
# Van Herwijnen 2016
def e_Herwijnen_2016(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    try:
        factor = scaling[0] / 0.93*scaling[1]**2.8
        return(factor * 0.93*rho**2.8)
    except: 
        print('no scaling used')
        return(0.93*rho**2.8)  
e_Herwijnen_2016_with_unc = uncertainties.wrap(e_Herwijnen_2016)

# Scapozza 2004 formel 6.4
def e_scapozza_2004(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    try:
        factor = scaling[0] / (0.1873 * np.exp(0.0149*scaling[1]))
        return(factor * 0.1873 * np.exp(0.0149*rho)* 1E6)
    except: 
        return(0.1873 * np.exp(0.0149*rho)* 1E6)
e_scapozza_2004_with_unc = uncertainties.wrap(e_scapozza_2004)

# Sigrist 2006 Diss formel 4.8 
def e_sigrist_2006(rho, **kwargs):
    ''' input: rho, [scaling = (Elastic modulus measured (MPa), density mesured), optional]
    output: elastic modulus (Pa) 
    '''
    scaling = kwargs.get('scaling', None)
    try:
        factor = scaling[0] / 1.89E-6*scaling[1]**2.94
        return(factor * 1.89E-6*rho**2.94* 1E6)
    except: 
        return(1.89E-6*rho**2.94* 1E6)

e_sigrist_2006_with_unc = uncertainties.wrap(e_sigrist_2006)

    
    
    