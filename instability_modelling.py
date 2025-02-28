# -*- coding: utf-8 -*-
"""
#%% Routines for instability modelling
@author: Bergfeld Bastian
"""

import sys
import numpy as np
import pandas as pd
import functools
import logging

sys.path.append('D:\\SMP_stability\\weac')
import weac as weac
sys.path.append('D:\\SMP_stability')
from logging_class import LoggerConfig, error_handling_decorator
    

class R2015_point_instability:
    def __init__(self, profile, skier_stability_params=None, PST_params=None):
        """
        Initializes the class with a layered snow profile.
        
        Parameters:
        - file_path: str, path to the layered snow profile pickle file
        - skier_stability_params: dict, to initialize weac:
            - totallength: float, lateral length of simulated snowpack in mm
            - skierweight: float, skier weight in kg
            - inclination: float, slope angle in degrees, negative clockwise
            - slab_load: bool, True or False matter if the stress, induced by the overlying slab, should be accounted for
        - PST_params: dict, to initialize weac:
            - system: string, slope normal beam ends '-pst', 'pst-' or vertial '-vpst', 'vpst-'; minus is cutting direction
            - totallength: float, lateral length of simulated snowpack in mm
            - inclination: float, slope angle in degrees, negative clockwise
            - max_cracklength: float, maximal crack length for which Energy release is computed in mm
            - num_da: float, number of infeniesimal crack increments within max_cracklength

        """
        self.file_path = profile.attrs["filepath"]
        self.skier_stability_params = skier_stability_params or {
            "totallength": 4e4, "inclination": -38, "skierweight": 80, "slab_load": True}
        self.PST_params = PST_params or {
            "system": "pst-", "totallength": 4e3, "inclination": -38, "max_cracklength": 2e3, "num_da": 500}
        self.profile = profile.copy()
        self.stab = self.profile[["depthTop","thickness"]].copy()

    @error_handling_decorator
    def _get_slab_profile_for_weac(self, wl_id):
        """Loads the snow profile from a pickle file and computes additional parameters."""
        density = self.profile["CR2020_density"].values
        thickness = self.profile["thickness"].values
        slab_profile = np.column_stack((density, thickness))[:wl_id, :].tolist()
        return slab_profile

#---------- S computation --------------------------------    
    @error_handling_decorator
    def _compute_tau(self, weak_layer_id):
        """Computes shear stress along a given weak layer."""
        skier = weac.Layered(system='skier', layers=self._get_slab_profile_for_weac(weak_layer_id))
        seg_skier = skier.calc_segments(L=self.skier_stability_params["totallength"], m=self.skier_stability_params["skierweight"])['nocrack']
        C_skier = skier.assemble_and_solve(phi=self.skier_stability_params["inclination"], **seg_skier)
        xsl_skier, z_skier, xwl_skier = skier.rasterize_solution(C=C_skier, phi=self.skier_stability_params["inclination"], **seg_skier)
        x, tau = skier.get_weaklayer_shearstress(xwl_skier, z_skier, unit="kPa")
        return(x,tau)
    @error_handling_decorator    
    def _compute_max_tau(self, weak_layer_id):
        """Computes max shear stress for a given weak layer without the static stress induce by the slab"""
        x, tau = self._compute_tau(weak_layer_id)
        # Compute maximal shear stress in the weak layer max_tau_skier
        if self.skier_stability_params["slab_load"]: # take the full shear stress in the weak layer
            tau_skier = tau
        else: # take just the additional shear stress induced by the skier
            tau_skier = tau - self.profile["load_above"].iloc[weak_layer_id] / 1e3 * np.sin(np.deg2rad(self.skier_stability_params["inclination"]))
        return max(abs(tau_skier))
    @error_handling_decorator
    def compute_skier_stability_S(self):
        """Computes the stability ratio S_Reuter2015 and adds it to the DataFrame."""
        logging.info(f"   compute_skier_stability_S for {self.profile.attrs["name"]} {self.profile.shape[0]} layers")

        max_tau_skier = self.profile.index.to_series().apply(self._compute_max_tau)
        self.stab["S_Reuter2015"] = self.profile["JS1999_sigma_macro"] * 1e3 / max_tau_skier

#---------- rc computation --------------------------------    
    @error_handling_decorator
    def _compute_rc_layer(self, weak_layer_id):
        """Computes rc for a given weak layer."""
        pst = weac.Layered(system=self.PST_params["system"], layers=self._get_slab_profile_for_weac(weak_layer_id))
    
        # Initialize outputs and crack lengths
        Gdif = np.zeros([3, self.PST_params["num_da"]])
        da = np.linspace(1e-6, self.PST_params["max_cracklength"], num=self.PST_params["num_da"])
        
        # Loop through crack lengths
        for i, a in enumerate(da):
            # Obtain lists of segment lengths, locations of foundations.
            seg_err = pst.calc_segments(L=self.PST_params["totallength"], a=a)
            # Assemble system and solve for free constants
            C1 = pst.assemble_and_solve(phi=self.PST_params["inclination"], **seg_err['crack']) 
            # Compute differential and incremental energy release rates
            Gdif[:, i] = pst.gdif(C1, self.PST_params["inclination"], **seg_err['crack'])
        
        w_f = self.profile["R2015_wf"][weak_layer_id]
        r_c = da[Gdif[0, :]*1000 > w_f].min()/10
        return(r_c)
        
    @error_handling_decorator
    def compute_critical_cut_length_rc(self):
        """Computes rc for a given weak layerand adds it to the DataFrame."""
        logging.info(f"   compute_critical_cut_length_rc for {self.profile.attrs["name"]} {self.profile.shape[0]} layers")
        self.stab["rc_Reuter2015"] = self.profile.index.to_series().apply(self._compute_rc_layer)

    @classmethod
    def run(cls, profile, skier_stability_params=None, PST_params=None):
        """Creates an instance and runs all necessary computations."""
        logging.info("starting R2015_point_instability...")
        instance = cls(profile, skier_stability_params, PST_params)
        instance.compute_skier_stability_S()
        instance.compute_critical_cut_length_rc()        
        return instance       