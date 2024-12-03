import os
import pandas as pd
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

TRESHOLD = 14

# Parameters
PATH = 'data/subdata_extended'
SAVING_PATH = f"data/subdata_extended_corrupted_{TRESHOLD}km"
PAIRWISE_DISTANCES_FILE = 'data/pairwise_distances.csv'

NOISE_MEAN = 0
NOISE_STD = 0.5

# Ensure the saving directory exists
os.makedirs(SAVING_PATH, exist_ok=True)

# Load pairwise distances
pairwise_distances = pd.read_csv(PAIRWISE_DISTANCES_FILE)

# Filter connections below the threshold
pairwise_distances_filtered = pairwise_distances[pairwise_distances['distance'] <= TRESHOLD]

# Get unique latitude and longitude pairs
unique_lat_lon1 = np.unique(pairwise_distances_filtered[['lat1', 'lon1']].values, axis=0)
unique_lat_lon2 = np.unique(pairwise_distances_filtered[['lat2', 'lon2']].values, axis=0)
combined_unique_lat_lon = np.concatenate((unique_lat_lon1, unique_lat_lon2), axis=0)
combined_unique_lat_lon = np.unique(combined_unique_lat_lon, axis=0)

# List of columns to apply noise
feature_columns = [
    'solar_zenith_angle', 'solar_azimuth_angle', 'surf_air_temp_masked',
    'surf_temp_masked', 'surf_spec_hum_masked', 'h2o_vap_tot_masked',
    'cloud_liquid_water_masked', 'atmosphere_mass_content_of_cloud_ice_masked'
]

# Function to apply noise to matching rows
def apply_noise_to_matching_rows(df, combined_unique_lat_lon, feature_columns, mean, std):
    print(f"Applying noise to {len(combined_unique_lat_lon)} unique lat, lon pairs.")
    for lat, lon in combined_unique_lat_lon:
        mask = (df['lat'] == lat) & (df['lon'] == lon)
        noise = np.random.normal(mean, std, (mask.sum(), len(feature_columns)))
        df.loc[mask, feature_columns] += noise
    return df

# Function to create PyTorch Geometric node features
def create_node_features(df, feature_columns):
    features = df[feature_columns].to_numpy()
    return torch.tensor(features, dtype=torch.float)

# Function to process a single file
def process_file(file_name):
    if not file_name.endswith('.csv'):
        return

    file_path = os.path.join(PATH, file_name)
    output_csv_path = os.path.join(SAVING_PATH, file_name)
    output_features_path = os.path.join(SAVING_PATH, file_name.replace('.csv', '.pt').replace('data', 'node_features'))

    # Load data
    df = pd.read_csv(file_path)
    
    # Apply noise to DataFrame
    df_with_noise = apply_noise_to_matching_rows(df, combined_unique_lat_lon, feature_columns, NOISE_MEAN, NOISE_STD)
    
    # Save the modified DataFrame
    df_with_noise.to_csv(output_csv_path, index=False)
    
    # Create and save node features
    features = create_node_features(df_with_noise, feature_columns)
    torch.save(features, output_features_path)

    # Create and save real features
    # real_features = create_node_features(df, feature_columns)
    # real_output_path = os.path.join(PATH, file_name.replace('.csv', '.pt').replace('data', 'node_features'))
    # torch.save(real_features, real_output_path)

    print(f"Processed and saved: {file_name}")

# List all files in the directory
all_files = os.listdir(PATH)

# Use multiprocessing to process files in parallel
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(process_file, all_files)
