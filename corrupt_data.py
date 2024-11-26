import re
import math
import pickle
import random
import glob
import os
import json
from itertools import combinations, chain

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import plotly.graph_objs as go
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

import networkx as nx
from geopy.distance import geodesic

import concurrent.futures
from multiprocessing import Pool

# Fix seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Create node features for the graph
def create_node_features(df):
    """
    Create PyTorch Geometric node features.
    """
    node_features = []
    node_mapping = {}
    index = 0

    for _, row in df.iterrows():
        node_id = (row['lat'], row['lon'])
        if node_id not in node_mapping:
            node_mapping[node_id] = index
            index += 1

        # Collect node features
        feature_vector = torch.tensor([
            row['solar_zenith_angle'],
            row['solar_azimuth_angle'],
            row['surf_air_temp_masked'],
            row['surf_temp_masked'],
            row['surf_spec_hum_masked'],
            row['h2o_vap_tot_masked'],
            row['cloud_liquid_water_masked'],
            row['atmosphere_mass_content_of_cloud_ice_masked']
        ], dtype=torch.float)
        node_features.append(feature_vector)

    return torch.stack(node_features), node_mapping

# Data corruption function
def corrupt_data(df, variables, noise_std=0.5):
    """
    Add Gaussian noise to specified variables in the dataframe.
    """
    corrupted_df = df.copy()
    for var in variables:
        corrupted_df[var] += np.random.normal(0, noise_std, df.shape[0])
    return corrupted_df

# Process data for a specific day
def process_day(df, nodes, year, month, day, combination, path):
    """
    Filter, process, and save daily data and corresponding node features.
    """
    df_filtered = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]
    df_filtered = df_filtered.merge(nodes, on=['lat', 'lon'], how='right')

    # Handle numeric columns by averaging duplicates
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns
    df_filtered = df_filtered.groupby(['lat', 'lon'], as_index=False)[numeric_cols].mean()

    # Fill missing values
    df_filtered.fillna(df_filtered.mean(), inplace=True)

    # Save processed data
    csv_path = os.path.join(path, f'data_{year:02}_{month:02}_{day:02}.csv')
    df_filtered.to_csv(csv_path, index=False)

    # Create and save node features
    x, _ = create_node_features(df_filtered)
    torch_path = os.path.join(path, f'node_features_{year:02}_{month:02}_{day:02}.pt')
    torch.save(x, torch_path)

    return f"Processed data for {year}-{month:02}-{day:02}. Node features shape: {x.shape}"

# Process combinations of variables
def process_combination(args):
    """
    Process a specific combination of variables with corruption and save results.
    """
    df, nodes, variables, combination = args
    combination_name = '_'.join(map(str, combination))
    path = os.path.join('data/subdata_extended_corrupted', combination_name)
    os.makedirs(path, exist_ok=True)

    # Corrupt data
    corrupted_df = corrupt_data(df, combination)

    # Process daily data
    results = []
    for year in corrupted_df['year'].unique():
        for month in corrupted_df[corrupted_df['year'] == year]['month'].unique():
            for day in corrupted_df[(corrupted_df['year'] == year) & (corrupted_df['month'] == month)]['day'].unique():
                result = process_day(corrupted_df, nodes, year, month, day, combination, path)
                results.append(result)
                print(result)

    return results

# Parallelize corruption and processing
def parallelize_corruption(df, nodes, variables, target_combinations):
    """
    Parallelize corruption and processing of data using multiprocessing for specific combinations.
    """
    combinations_tasks = [
        (df, nodes, variables, [variables[i] for i in combination])  # Map indices to variable names
        for combination in target_combinations
    ]

    with Pool() as pool:
        pool.map(process_combination, combinations_tasks)

# Main execution
if __name__ == "__main__":
    df = pd.read_csv('data/data_preprocessed_2013_2015.csv')

    # Define variables for processing
    variables = [
        'solar_zenith_angle',
        'solar_azimuth_angle',
        'surf_air_temp_masked',
        'surf_temp_masked',
        'surf_spec_hum_masked',
        'h2o_vap_tot_masked',
        'cloud_liquid_water_masked',
        'atmosphere_mass_content_of_cloud_ice_masked'
    ]

    # Normalize variables
    for var in variables:
        df[var] = (df[var] - df[var].mean()) / df[var].std()

    # Round coordinates and merge with unique nodes
    df['lat'] = df['lat'].round(1)
    df['lon'] = df['lon'].round(1)
    nodes = pd.read_csv('data/unique_coords.csv')
    df = df.merge(nodes, on=['lat', 'lon'], how='right')

    # Target combinations
    target_combinations = [
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3],
        [1, 7],
        [3]
    ]

    # Execute parallel corruption and processing
    parallelize_corruption(df, nodes, variables, target_combinations)
