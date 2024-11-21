
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

# FIX SEED
torch.manual_seed(1)
np.random.seed(1)
from multiprocessing import Pool

def create_node_features(df):
    # Initialize lists for PyTorch Geometric
    node_features = []
    
    # Step 1: Create a dictionary to map each (lat, lon, day, hour) combination to a node index
    node_mapping = {}
    index = 0
    for _, row in df.iterrows():
        node_id = (row['lat'], row['lon'])
        node_mapping[node_id] = index

        # Collect node features as a list of feature vectors
        feature_vector = torch.tensor([
            row['solar_zenith_angle'],
            row['solar_azimuth_angle'],
            # row['air_temp_masked'][0],
            row['surf_air_temp_masked'],
            row['surf_temp_masked'],
            row['surf_spec_hum_masked'],
            row['h2o_vap_tot_masked'],
            row['cloud_liquid_water_masked'],
            row['atmosphere_mass_content_of_cloud_ice_masked']
        ], dtype=torch.float)
        
        node_features.append(feature_vector)
        index += 1

    # Convert node features to a tensor
    return torch.stack(node_features), node_mapping

#### Data corruption
def corrupt_data(df, variables, noise_std=0.5):
    corrupted_df = df.copy()
    for var in variables:
        corrupted_df[var] = df[var] + np.random.normal(0, noise_std, df.shape[0])
    return corrupted_df

def process_day(df, nodes, year, month, day, combination, path):
    """Process data for a specific day."""
    # Filter data for the specific day
    df_filtered = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]
    df_filtered = df_filtered.merge(nodes, on=['lat', 'lon'], how='right')
    
    # Handle numeric columns by computing the mean for duplicates
    df_filtered_numeric = df_filtered.select_dtypes(include=['number']).groupby(['lat', 'lon'], as_index=False).mean()
    
    # Merge back non-numeric columns
    df_filtered = df_filtered[['lat', 'lon']].drop_duplicates().merge(df_filtered_numeric, on=['lat', 'lon'], how='left')
    
    # Fill NA values with column means
    for col in df_filtered.select_dtypes(include=['number']).columns:
        df_filtered[col].fillna(df_filtered[col].mean(), inplace=True)
    
    # Save to CSV
    csv_path = f'{path}/data_{year:02}_{month:02}_{day:02}.csv'
    df_filtered.to_csv(csv_path, index=False)
    
    # Create and save node features
    x, _ = create_node_features(df_filtered)
    torch_path = f'{path}/node_features_{year:02}_{month:02}_{day:02}.pt'
    torch.save(x, torch_path)
    
    return f"Data for {year}-{month}-{day} has been processed. Shape: {x.shape}"


def process_combination(args):
    """Process a specific combination of variables."""
    df, nodes, variables, combination = args
    name = '_'.join(combination)
    path = f'corruption2/{name}'
    os.makedirs(path, exist_ok=True)

    # Corrupt data
    corrupted_df = corrupt_data(df, combination)

    # Process days sequentially
    results = []
    for year in corrupted_df['year'].unique():
        for month in corrupted_df[corrupted_df['year'] == year]['month'].unique():
            for day in corrupted_df[(corrupted_df['year'] == year) & (corrupted_df['month'] == month)]['day'].unique():
                result = process_day(corrupted_df, nodes, year, month, day, combination, path)
                results.append(result)
                print(result)

    return results


def parallelize_corruption(df, nodes, variables):
    """Parallelize the corruption and processing of data."""
    combinations_tasks = [
        (df, nodes, variables, combination)
        for i in range(1, 8)
        for combination in combinations(variables, i)
    ]
    
    # Use a pool of workers to process each combination
    with Pool() as pool:
        pool.map(process_combination, combinations_tasks)


# Example call
# parallelize_corruption(df, nodes, variables)




df=pd.read_csv('data/data_preprocessed.csv')

# Normalize all the variables
variables = ['solar_zenith_angle', 
             'solar_azimuth_angle', 
             'surf_air_temp_masked', 
             'surf_temp_masked', 
             'surf_spec_hum_masked', 
             'h2o_vap_tot_masked', 
             'cloud_liquid_water_masked', 
             'atmosphere_mass_content_of_cloud_ice_masked']

#### Feature correlation
correlation_matrix = df[variables].corr()

#### Feature extraction
# Round lan and lon to 1 decimal
df['lat'] = df['lat'].round(1)
df['lon'] = df['lon'].round(1)

nodes = pd.read_csv('data/unique_coords.csv')

# Merge the nodes with the df
df=df.merge(nodes, on=['lat', 'lon'], how='right')
for var in variables:
    df[var] = (df[var] - df[var].mean()) / df[var].std()

# Example call
parallelize_corruption(df, nodes, variables)
