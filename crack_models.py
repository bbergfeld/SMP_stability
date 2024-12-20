# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:46:20 2020
#%% theoretical prediction of crack speed and elastic wave speeds in snow
@author: gerling
"""


import os
import xarray as xr
import numpy as np
os.chdir('N:\\lawprae\\LBI\\Projects\\206_Dynamic_crack_propagation\\postprocessing-scripts\\DIC_PROCESSING')
from functions import *
from PST_class import *
from uncertainties import ufloat, unumpy
import uncertainties
from uncertain_panda import pandas as pd


# Theoretical estimates of weak layer crack speeds in snow:

def solitary_wave_speed(g, E, h, hf, rho):
    ''' Heierli 2005 - Solitary fracture waves in metastable snow stratifications, Journal of Geophysical Research, Equation 7a
    g = gravitational constant
    E = elastic modulus slab
    h = slab thickness
    hf = collapse height
    rho = mean slab density
    '''
    D = (E*h**3)/12 # flexural rigidity of the slab 
    c4 = (g*D)/(2*hf*rho*h)
    c = unumpy.sqrt(unumpy.sqrt(c4))
    d = { 'modelled_speed': c}
    df = pd.DataFrame(data=d,index=E.index)   
    df.modelled_speed.unit = 'm s$^{-1}$'
    return(df)
solitary_wave_speed_with_unc = uncertainties.wrap(solitary_wave_speed)

def solitary_wave_touchdown(g, E, h, hf, rho):
    ''' Heierli 2005 - Solitary fracture waves in metastable snow stratifications, Journal of Geophysical Research, Equation 7a
    g = gravitational constant
    E = elastic modulus slab
    h = slab thickness
    hf = collapse height
    rho = mean slab density
    '''
    gamma = 2.331 # from Heierli
    D = (E*h**3)/12 # flexural rigidity of the slab 
    tdd4 = gamma**4 * (2*hf*D)/(g*rho*h)
    tdd = unumpy.sqrt(unumpy.sqrt(tdd4))
    d = { 'modelled_touchdown': tdd}
    df = pd.DataFrame(data=d,index=E.index)   
    df.modelled_touchdown.unit = 'm'
    return(df)
solitary_wave_touchdown_with_unc = uncertainties.wrap(solitary_wave_touchdown)

def mc_clung_fracture_speeds (nu, E, rho):
    ''' Mc Clung 2005 - Approximate estimates of fracture speeds for dry slab avalanches, Geophysical Research letters, Equation 1
    E = elastic modulus slab
    nu =  Poisson ratio (-)
    rho = mean slab density
    '''
    G = E/(2*(1+nu))
    low_c = 0.7*np.sqrt(G/rho)
    high_c = 0.9*np.sqrt(G/rho)
    return(low_c,high_c)
    
def anticrack_propagation_speed(g, E, nu, h, hf, rho, theta, l, C_initial_guess=0.5):
    ''' Heierli dissertation - Anticrack model for slab avalanche release, Equation 5.17
        the formula is a dispersion relation which links speed ot touchdowndistance
        hf = sollapse height (m)
        rho = slab mean density (kg/m^3)
        h = slab thickness (m)
        theta = slope angle (°)
        nu = poisson Ratio (-)
        E = slab elastic modulus (Pa)
        l = touchdown distance (m)
    returns crack propagation speed in m/s             
    
    --Note: computation is sensitive to the C_initial_guess -- 
    '''
    from scipy import  optimize

    def get_speed_for_td_length(L):  
        func = lambda C : (L**2/C**2)*((eta/(L*C*(1-C**2))) * (1/np.sin(2*C*L/eta)-1/np.tan(2*C*L/eta))-1)- 2*H_f/Sigma
        C_solution = optimize.fsolve(func, C_initial_guess)
        return(C_solution)
            
    #% values as Heierli page 68
    theta = np.radians(theta)
    k = 5/6 # timoshenko beam correction factor for rectengular beam
    G = E/(2*(1+nu)) # shear modulus 
    eta = np.sqrt(E/(3*k*G))
    
    shear_wave_velocity = np.sqrt(k*G/rho)
    longitudinal_wave_velocity = np.sqrt(E/rho)
    #compressive stress
    sigma = -rho*g*h*np.cos(theta) 
    #shear stress
    tau = rho*g*h*np.sin(theta)
    #dimensionless variables  Table 4.1 Heierli diss and Equ. 5.2
    L = l/h
    H_f = hf/h  # collapse amplitude
    Sigma = -sigma/(k*G) # dimensionless compressive stress // Table 4.1 Heierli diss
    Tau = tau/(k*G)# dimensionless compressive stress // Table 4.1 Heierli diss

        
    C = get_speed_for_td_length(L)
    c = C*shear_wave_velocity
    return(c) 


# Theoretical estimates for wave speeds in materials:

def shear_wave_speed(G, rho):
    ''' 
    G = shear modulus slab
    rho = mean slab density
    '''
    return(np.sqrt(G/rho))
    
def long_wave_speed(E, rho):
    return(np.sqrt(E/rho))

def rayleigh_wave_speed(E, G, rho,nu):
    ''' Bergmann approximation
    E = Elastic modulus of slab (Pa)
    G = shear modulus slab  (Pa)
    rho = mean slab density (kg/m3)
    nu = Poisson Ratio of slab
    '''
    x = np.sqrt((0.87 + 1.12*nu)/(1+nu))
    return(x * shear_wave_speed(G,rho,))
    
def youngs_to_pwave_modulus(E,nu):
    return((E*(1-nu))/((1+nu)*(1-2*nu)))


    
