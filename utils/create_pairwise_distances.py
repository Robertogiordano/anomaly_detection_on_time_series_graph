import pandas as pd
from geopy.distance import geodesic
import itertools
import concurrent.futures

# Load coordinates
coords = pd.read_csv('data/unique_coords.csv')

# Extract lat/lon pairs
lat_lon_pairs = coords[['lat', 'lon']].values

# Function to calculate the distance for a pair of coordinates
def calculate_distance(pair):
    """
    Calculate the geodesic distance for a given pair of coordinates.
    
    Parameters:
        pair (tuple): A tuple containing two coordinate pairs ((lat1, lon1), (lat2, lon2)).
    
    Returns:
        dict: A dictionary with lat1, lon1, lat2, lon2, and the distance.
    """
    (lat1, lon1), (lat2, lon2) = pair
    print(f"Calculating distance between ({lat1}, {lon1}) and ({lat2}, {lon2})")
    dist = geodesic((lat1, lon1), (lat2, lon2)).km
    return {'lat1': lat1, 'lon1': lon1, 'lat2': lat2, 'lon2': lon2, 'distance': dist}

# Generate all unique pairs of coordinates
coordinate_pairs = list(itertools.combinations(lat_lon_pairs, 2))

# Use ThreadPoolExecutor for parallel computation
def calculate_distances_parallel(coordinate_pairs):
    """
    Calculate distances for all coordinate pairs in parallel.
    
    Parameters:
        coordinate_pairs (list): List of coordinate pairs.
    
    Returns:
        DataFrame: A DataFrame containing lat1, lon1, lat2, lon2, and the distance.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the distance calculation function to coordinate pairs
        for result in executor.map(calculate_distance, coordinate_pairs):
            results.append(result)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

# Perform the calculation
distance_df = calculate_distances_parallel(coordinate_pairs)

# Save the results
distance_df.to_csv('data/pairwise_distances.csv', index=False)

print("Distance DataFrame created with shape:", distance_df.shape)
