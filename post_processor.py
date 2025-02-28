# -*- coding: utf-8 -*-
"""
#%% Routines for visualization and combining of instability metrics
@author: Bergfeld Bastian
"""
import logging
import sys
import os
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

    
def compute_harmonic_mean(instability_instance, metric_1, metric_2):
    I1,I2 = instability_instance.stab[metric_1], instability_instance.stab[metric_2]
    instability_instance.stab["harmonic_mean"] = (2*I1*I2) / (I1+I2)
def compute_logarithmic_mean(instability_instance, metric_1, metric_2):
    I1,I2 = instability_instance.stab[metric_1], instability_instance.stab[metric_2]
    instability_instance.stab["logarithmic_mean"] = (I1-I2) / (np.log(I1)-np.log(I2))
def compute_geometric_mean(instability_instance, metric_1, metric_2):
    I1,I2 = instability_instance.stab[metric_1], instability_instance.stab[metric_2]
    instability_instance.stab["geometric_mean"] = np.sqrt(I1*I2)
def compute_reziprocal_sum(instability_instance, metric_1, metric_2):
    I1,I2 = instability_instance.stab[metric_1], instability_instance.stab[metric_2]
    instability_instance.stab["reziprocal_sum"] = 1/(1/I1+1/I2)
def compute_logarithmic_sensitivity(instability_instance, metric_1, metric_2):
    I1,I2 = instability_instance.stab[metric_1], instability_instance.stab[metric_2]
    instability_instance.stab["logarithmic_sensitivity"] = np.exp(np.log(I1)+np.log(I2))


class plotter:
    """Handles standardized plotting for instances of th instability class."""

    @staticmethod 
    def density_vs_sth(instability_instance, ax1_metric = "CR2020_density", ax2_metric = "rc_Reuter2015", ax1=None, padding = 1.8):
        """
        Plots two metrics vs. depth for a given instability instance.

        Parameters:
        - model: instability_modelling instance
        - ax1_metric: standard is Density
        - ax2_metric: instability metric (e.g. rc_Reuter2015)
        - ax1: optional to plot in a given axis
        - padding: to seperate the left and right profile
        Returns:
        - ax1, ax2: Axes objects for further customization
        """
        def normalize(data, padding, percentile = 90):
            return(data / data.max(), np.percentile(data.dropna(), percentile) * padding/ data.max())
        df1 = instability_instance.profile
        df2 = instability_instance.stab
        df = pd.concat([df1, df2.drop(columns=df1.columns.intersection(df2.columns))], axis=1).copy()

        ax1_color = "lightblue"
        ax2_color = "red"
        df["depthBottom"] = df["depthTop"] + df["thickness"]
        ax1_data, ax1_xmax = normalize(df[ax1_metric], padding)
        ax2_data, ax2_xmax = normalize(df[ax2_metric], padding)
        
        if ax1 is None:
            fig, ax1 = plt.subplots(figsize=(6, 8))
            ax2 = ax1.twiny()
        else:
            fig = ax1.figure  # Use the figure from ax1 if provided
            if ax2:
                pass
            else: ax2 = ax1.twiny()

        # Plot as blocks
        for i, row in df.iterrows():
            ax1.fill_betweenx(
                [row["depthTop"], row["depthBottom"]],
                0, ax1_data.iloc[i],
                color=ax1_color, edgecolor='black', linewidth=1, alpha=0.6 )
            ax2.fill_betweenx(
                [row["depthTop"], row["depthBottom"]],
                0, ax2_data.iloc[i],
                color=ax2_color, edgecolor='black', linewidth=1, alpha=0.6 )
    
        # Labels & Formatting
        ax1.set_xlabel(ax1_metric, color=ax1_color)
        ax1.set_ylabel("Depth (mm)")
        ax1.invert_yaxis()
        ax1.grid(True, linestyle="--", alpha=0.2)
        ax1.set_xlim(0, ax1_xmax)
        ax1.set_ylim(df["depthBottom"].max()*1.05, -50)
        ax1.tick_params(axis='x', colors=ax1_color)
        ax1.grid(True, linestyle="--", alpha=0.2)

        ax2.set_xlabel(ax2_metric, color="red")
        ax2.set_xlim(0, ax2_xmax)  # Ensure full density range + padding
        ax2.invert_xaxis()  # Depth increases downward
        ax2.tick_params(axis='x', colors=ax2_color)
        # Manually create legend elements
        ax1_dummy_line = ax1.plot([], [], color=ax1_color, label=ax1_metric)
        ax1.legend(handles=ax1_dummy_line, loc = "upper left")
        ax2_dummy_line = ax2.plot([], [], color=ax2_color, label=ax2_metric)
        ax2.legend(handles=ax2_dummy_line, loc = "upper right")        
        return ax1, ax2

    


    @classmethod
    def run(cls, instance):
        """Creates an instance and runs all necessary computations."""
        logging.info("starting PostProcessor...")
        compute_harmonic_mean(instance, "S_Reuter2015", "rc_Reuter2015")
        compute_logarithmic_mean(instance, "S_Reuter2015", "rc_Reuter2015")
        compute_geometric_mean(instance, "S_Reuter2015", "rc_Reuter2015")
        compute_reziprocal_sum(instance, "S_Reuter2015", "rc_Reuter2015")
        compute_logarithmic_sensitivity(instance, "S_Reuter2015", "rc_Reuter2015")
        plotter.density_vs_sth(instance, ax2_metric="logarithmic_sensitivity")  
        return instance       