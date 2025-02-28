# -*- coding: utf-8 -*-
"""
#%% Routines to process a pnt file in order to do mechanical calculation on the measured snow profile
@author: Bergfeld Bastian
"""

import sys
import os
import logging
import pickle
import concurrent.futures
os.environ["OMP_NUM_THREADS"] = "3" # nur bei windows, memory leak durch KMeans-Implementierung in scipy
sys.path.append('D:\\SMP_stability\\snowmicropyn')
import snowmicropyn as smpyn
from snowmicropyn.serialize import caaml
from snowmicropyn.ai.grain_classifier import grain_classifier
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.integrate import cumulative_trapezoid
import functools

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

sys.path.append('D:\\SMP_stability\\weac')
import weac as weac
from Emod_parametrizations import e_gerling_2017_AC
import itertools
sys.path.append('D:\\SMP_stability')
from logging_class import LoggerConfig, error_handling_decorator


class PreProcessor:
    """- Loads *.pnt file from a source.
       - get clusters
       - get mechanical properties"""
    
    def __init__(self, source):
        """Handles input validation: single file, list, or folder."""
        if isinstance(source, str):  
            if os.path.isfile(source) and source.lower().endswith('.pnt'):
                self.source = source  # ✅ Store as a string for a single file
            elif os.path.isdir(source):  
                self.source = self._get_pnt_files_from_folder(source)  # Returns a list
            else:
                raise ValueError("Error: Provided string is not a valid .pnt file or folder.")
        
        elif isinstance(source, list):  
            if all(isinstance(f, str) and f.lower().endswith('.pnt') for f in source):
                self.source = source  # ✅ Store as a list for multiple files
            else:
                raise ValueError("Error: List contains non-.pnt files or invalid elements.")
        else:
            raise TypeError("Error: Input should be a string (file path or folder) or a list of .pnt files.")

    def _get_pnt_files_from_folder(self, folder):
        """Returns a list of all .pnt files in the provided folder."""
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pnt')]




    @error_handling_decorator
    def _load_pnt_file(self, source: str) -> pd.DataFrame | None:
        """Loads a .pnt file.
        Input:
            source (str): Path to the .pnt file.
        """
        prof = smpyn.Profile.load(source)
        return prof.samples_within_snowpack()


    @error_handling_decorator
    def _calc_basic_features(self, overlap: float = 50, window_size: float = 2, parameterization: str = "CR2020") -> pd.DataFrame | None:
        """computes microstructural parameters (Loewe et al. 2012), density and SSA, and weak layer fracture energy
    
        Input:
            overlap (float, optional): Overlapping percentage of windows (default: 50).
            windowsize (float, optional): Size of window to compute microstructural paramters (default: 2mm).
            parameterization (str, optional): Parameterization method for density and SSA (default: "CR2020").
        """  
        samples = self._load_pnt_file(self.source)
        param = smpyn.derivatives.parameterizations[parameterization] # Prepare derivatives:
        param.overlap = overlap
        param.window_size = window_size
        loewe2012_df = smpyn.loewe2012.calc(samples, param.window_size, param.overlap)
        derivatives = loewe2012_df
        derivatives = derivatives.merge(param.calc_from_loewe2012(loewe2012_df))
        derivatives.profile_bottom = derivatives.distance.iloc[-1]
        attrs = {"name":os.path.basename(self.source), "filepath":os.path.abspath(self.source), "overlap": overlap, "window_size": window_size, "parameterization": parameterization}
        derivatives.attrs = attrs
        return derivatives

    @error_handling_decorator
    def _get_layers(self, derivatives: pd.DataFrame, additional_clusters=0, feature_weights=None) -> pd.DataFrame | None:
        """Performs layer clustering using KMeans to detect weak layers.
    
        Input:
            derivatives (pd.DataFrame): Processed derivatives from PreProcessor._load_pnt_file.
            additional_clusters (int): Additional clusters to allow for weak layer detection.
            feature_weights (dict, optional): Custom weights for each feature.
    
        Output:
            pd.DataFrame or None: DataFrame with added cluster and layer information.
        """    
        # Define default features and their weights
        default_features = {
            "force_median": 3,
            "L2012_f0": 1,
            "L2012_delta": 1,
            "L2012_L": 3,
            "CR2020_ssa": 3,
            "CR2020_density": 1
        }
        # Use provided feature weights, otherwise default
        feature_weights = feature_weights if feature_weights is not None else default_features
        # Check if required features exist in DataFrame
        missing_features = [feat for feat in feature_weights.keys() if feat not in derivatives.columns]
        if missing_features:
            logging.error(f"Missing features in DataFrame: {missing_features}")
            return None 
        # Create feature matrix and apply weights
        features = np.column_stack([derivatives[feat].values for feat in feature_weights.keys()])
        weights = np.array(list(feature_weights.values()))
        scaler = StandardScaler() # Scale features and apply weights
        scaled_features = scaler.fit_transform(features) * weights
    
        # Determine optimal number of clusters using the elbow method
        min_cluster, max_cluster = 1, 20
        cluster_range = np.arange(min_cluster, max_cluster)
        wcss = []
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            wcss.append(kmeans.inertia_)
        # Compute maximum curvature to determine optimal clusters
        line_start, line_end = np.array([cluster_range[0], wcss[0]]), np.array([cluster_range[-1], wcss[-1]])
        distances = [np.abs(np.cross(line_end - line_start, line_start - np.array([cluster_range[i], wcss[i]]))) / np.linalg.norm(line_end - line_start)
            for i in range(len(cluster_range))]
        opt_number_clusters = cluster_range[np.argmax(distances)]
        n_clusters = opt_number_clusters + additional_clusters
    
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        derivatives["cluster_id"] = labels
    
        # Determine boundaries between clusters to get layering:
        x = derivatives["distance"].values
        boundaries = [x[0]] + [x[i] for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [x[-1]]
        derivatives["layer_id"] = (derivatives["cluster_id"] != derivatives["cluster_id"].shift()).cumsum()

        layered_derivatives = derivatives.groupby('layer_id').mean()
        layered_derivatives["distance"] = derivatives.groupby('layer_id').min().distance # distance is the lower boundary in the snowpack, 0 is surface
        layered_derivatives.profile_bottom = derivatives.profile_bottom
        layered_derivatives.index = layered_derivatives.index - 1
        layered_derivatives = layered_derivatives.rename(columns={'distance': 'depthTop'})
        layered_derivatives.profile_bottom = derivatives.profile_bottom

        thickness_values = []
        for idx, row in layered_derivatives.iterrows():
            if idx == len(layered_derivatives) - 1:
                layer_thickness = layered_derivatives.profile_bottom - row.depthTop
                thickness_values.append(layer_thickness)
            else:
                layer_thickness = layered_derivatives.depthTop[idx + 1] - row.depthTop
                thickness_values.append(layer_thickness)
        layered_derivatives['thickness'] = thickness_values
        layered_derivatives = layered_derivatives[['depthTop','thickness', 'cluster_id', 'force_median', 'L2012_lambda', 'L2012_f0',
                 'L2012_delta', 'L2012_L', 'CR2020_density', 'CR2020_ssa']] # Reorder the DataFrame columns

        logging.info(f"   {layered_derivatives.attrs['name']} - Optimal number of clusters: {opt_number_clusters}, using {n_clusters}.")
    
        return layered_derivatives


    @error_handling_decorator
    def _get_mechanical_properties(self, layered_derivatives: pd.DataFrame) -> pd.DataFrame | None:

        def _get_weak_layer_fracture_energy(layered_derivatives):
            def rolling_integral(df, window_size):
                half_window = window_size / 2
                distances = df['distance'].values
                forces = df['force'].values
                integrated_values = np.full(len(df), np.nan)
                
                # Use a sliding window approach
                left_idx = np.searchsorted(distances, distances - half_window, side='left')
                right_idx = np.searchsorted(distances, distances + half_window, side='right')
                
                for i in range(len(df)):
                    # Ensure full window coverage
                    if distances[right_idx[i] - 1] - distances[left_idx[i]] >= 0.99 * window_size:
                        integral = cumulative_trapezoid(forces[left_idx[i]:right_idx[i]], distances[left_idx[i]:right_idx[i]], initial=0)
                        integrated_values[i] = integral[-1]  # Take the full integral over the window
                
                return integrated_values
            
            # Load sample data
            df = self._load_pnt_file(self.source)
            window_size = 2  # mm
            df["w_f"] = rolling_integral(df, window_size)
            
            # Compute minimum w_f within each layer
            w_f = [
                df.loc[(df.distance > row.depthTop) & (df.distance < row.depthTop + row.thickness), "w_f"].min()
                for _, row in layered_derivatives.iterrows()
            ]
            
            # Store results in layered_derivatives
            layered_derivatives["R2015_wf"] = w_f
            layered_derivatives["R2015_wf"].attrs = {
                "unit": "mJ",
                "window_size": window_size,
                "info": "Implementation of the Fracture energy of Reuter et al 2015. Note that this has incorrect units."
            }
            return layered_derivatives
    
        layered_derivatives = _get_weak_layer_fracture_energy(layered_derivatives)
        
        # Compute additional material properties
        from Emod_parametrizations import e_gerling_2017_AC
        layered_derivatives['G2017_E_AC'] = layered_derivatives['CR2020_density'].apply(e_gerling_2017_AC)
        layered_derivatives.G2017_E_AC.attrs = {"unit":"N/m^2","info":"Pa - elastic modulus estimate through density parametrization"}

        layered_derivatives["JS1999_sigma_n"] = layered_derivatives.L2012_f0 / (layered_derivatives.L2012_L ** 2)
        layered_derivatives.JS1999_sigma_n.attrs = {"unit":"N/mm^2","info":"MPa - Microstructural element compressive strength"}

        layered_derivatives["JS1999_sigma_macro"] = layered_derivatives.JS1999_sigma_n * (layered_derivatives.L2012_delta / layered_derivatives.L2012_L)
        layered_derivatives.JS1999_sigma_macro.attrs = {"unit":"N/mm^2","info":"MPa - macroscale compressive strength"}
        
        # Compute layer weight (load above weak layer)
        layer_weight = layered_derivatives.CR2020_density * layered_derivatives.thickness/1000  # kg/m^2
        layer_weight = layer_weight.cumsum().shift(1, fill_value=0)  # Accumulate weight of layers above
        layer_weight = layer_weight * -9.81  # Convert to Pa
        layered_derivatives["load_above"] = layer_weight
        layered_derivatives.load_above.attrs['unit'] = "kg/m^2, Pa"

        self.df = layered_derivatives
        logging.info(f"   {layered_derivatives.attrs['name']} - Number of layers: {len(layered_derivatives)}")

        return layered_derivatives

    def plot_density(self):
        fig, ax = plt.subplots(figsize=(7, 8))
        # Compute depth by cumulative sum of thickness
        self.df["depth_top"] = self.df["thickness"].cumsum().shift(1, fill_value=0)
        self.df["depth_bottom"] = self.df["depth_top"] + self.df["thickness"]
        # Set density x-limit slightly beyond the max value
        density_xmax = self.df["CR2020_density"].max() * 1.1  # 10% padding
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 8))
        
        # Plot density as blocks
        for _, row in self.df.iterrows():
            ax.fill_betweenx(
                [row["depth_top"], row["depth_bottom"]],
                0, row["CR2020_density"],
                color='lightblue', edgecolor='black', linewidth=1, alpha=0.6
            )
        
        # Labels & Formatting
        ax.set_xlabel("Density (kg/m³)", color="blue")
        ax.set_ylabel("Depth (mm)")
        ax.invert_yaxis()  # Depth increases downward
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(0, density_xmax)  # Ensure full density range + padding
        ax.tick_params(axis='x', colors='blue')  # Color density axis for clarity

    def get_pnt_files(path):
        """
        Checks if the provided path is:
        1. A .pnt file -> Returns the filename.
        2. A folder -> Returns a list of all .pnt files in the folder.
        3. A list of .pnt files -> Returns the list as is.
        """
        if isinstance(path, list):  
            # Check if path is already a list of .pnt files
            if all(isinstance(f, str) and f.lower().endswith('.pnt') for f in path):
                return path  
            else:
                return "Error: The list contains non-.pnt files or invalid items."
    
        elif os.path.isfile(path):
            # Check if it's a .pnt file
            if path.lower().endswith('.pnt'):
                return [os.path.basename(path)]
            else:
                return f"Error: {path} is a file but not a .pnt file."
        
        elif os.path.isdir(path):
            # Get all .pnt files in the folder
            pnt_files = [f for f in os.listdir(path) if f.lower().endswith('.pnt')]
            return pnt_files if pnt_files else "No .pnt files found in the folder."
        
        else:
            return "Error: Invalid file, folder path, or list."

    @classmethod
    def run(cls, source):
        """Runs the file processing, collects DataFrames, and saves them as pickle files."""
        logging.info(f"Preprocessing started...")

        instance = cls(source)  # Create an instance for validation

        if isinstance(instance.source, str):  
            df = cls.run_file(instance.source)
            cls.save_dataframe(instance.source, df)
        else:
            # ✅ Multiple files case (parallel processing)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(cls.run_file, file): file for file in instance.source}
                
                for future in concurrent.futures.as_completed(futures):
                    file = futures[future]  # Get the associated file
                    try:
                        df = future.result()  # Get DataFrame result
                        cls.save_dataframe(file, df)  # Save DataFrame as pickle
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        logging.info(f"Preprocessing finished...")


    @classmethod
    def save_dataframe(cls, file, df):
        """Saves a DataFrame as a pickle file, modifying the filename."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Error: Expected a Pandas DataFrame.")
        new_filename = file[:-4] + "_clustered_profile.pkl"
        with open(new_filename, "wb") as f:
            pickle.dump(df, f)
        logging.info(f"   Saved: {new_filename}")
    
    @classmethod
    def run_file(cls, pnt_file):
        """Processes a single file (file is passed as a string)."""
        if not isinstance(pnt_file, str):
            raise TypeError(f"Expected a string file path, but got {type(pnt_file)}")

        profile = cls(pnt_file)
        pnt_prof = profile._calc_basic_features()
        lay_prof = profile._get_layers(pnt_prof)
        df_layers = profile._get_mechanical_properties(lay_prof)
        return df_layers       

