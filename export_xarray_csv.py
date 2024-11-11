import os
import pandas as pd
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_file(file, PATH, mapping):
    # List to collect row dictionaries
    print(f"Processing file: {file}")
    rows = []
    data = xr.open_dataset(os.path.join(PATH, file), engine='netcdf4')
    satellite_positions = {a: data[mapping['satellite_position']].sel(atrack=a).values for a in data['atrack'].values}

    # Loop through atrack and xtrack positions
    for a in data['atrack'].values:
        sat_pos = satellite_positions[a]
        for x in data['xtrack'].values:
            row = {'file': data.id + ' - ' + file, 'atrack': a, 'xtrack': x, 'satellite_position': sat_pos}
            
            for key, value in mapping.items():
                if key in {'file', 'satellite_position', 'atrack', 'xtrack'}:
                    continue  # Skip keys handled above
                
                if key.endswith('_masked'):
                    qc_key = mapping[key] + '_qc'
                    qc_value = data[qc_key].sel(atrack=a, xtrack=x).values
                    
                    # Handle masked values based on QC
                    if 2 in qc_value:
                        current_value = data[mapping[key]].sel(atrack=a, xtrack=x).values
                        if len(current_value.shape) == 0:  # scalar
                            row[key] = np.nan if qc_value == 2 else current_value
                        else:  # array
                            row[key] = np.where(qc_value == 2, np.nan, current_value)
                    else:
                        row[key] = data[mapping[key]].sel(atrack=a, xtrack=x).values
                else:
                    row[key] = data[mapping[key]].sel(atrack=a, xtrack=x).values

            # Add the populated row to rows list
            rows.append(row)

    return rows

def main():
    PATH='SNDRSNML2RMS_1-20241110_220957'
    mapping={
        'file': 'id', #file name
        'atrack': 'atrack', #along-track index
        'xtrack': 'xtrack', #cross-track index
        'date': 'obs_time_tai93', #time of observation in TAI93
        'lat': 'lat_geoid', #latitude of FOV center on the geoid (without terrain correction)
        'lon': 'lon_geoid', #longitude of FOV center on the geoid (without terrain correction)
        'surf_alt': 'surf_alt', #mean surface altitude wrt  earth model over the FOV
        'solar_zenith_angle': 'sol_zen', #solar zenith angle at the center of the spot
        'solar_azimuth_angle': 'sol_azi', #solar azimuth angle at the center of the spot
        'satellite_position': 'sat_pos', #satellite ECR position at scan_mid_time

        'air_temp_masked': 'air_temp', #list of different air temperature values at different pressure levels masked with qc

        'surf_air_temp_masked': 'surf_air_temp', #near-surface air temperature (~2 meters above surface) masked with qc
        'surf_temp_masked': 'surf_temp', #surface temperature masked with qc
        'surf_spec_hum_masked': 'surf_spec_hum', #Near-surface mass fraction of water vapor in moist air masked with qc
        'surf_spec_hum_masked': 'surf_spec_hum', #Near-surface mass fraction of water vapor in moist air masked with qc

        'h2o_vap_tot_masked': 'h2o_vap_tot', #total water vapor content masked with qc
        'cloud_liquid_water_masked': 'h2o_liq_tot', #total cloud liquid water content masked with qc
        'atmosphere_mass_content_of_cloud_ice_masked': 'h2o_ice_tot' #total cloud ice content masked with qc
    }

    files = os.listdir(PATH)

    # Use ProcessPoolExecutor for parallel file processing
    with ProcessPoolExecutor() as executor:
        # Prepare partial function to pass constant arguments
        process_partial = partial(process_file, PATH=PATH, mapping=mapping)
        
        # Process files in parallel
        all_rows = []
        for result in executor.map(process_partial, files):
            all_rows.extend(result)
    
    # Create DataFrame from all rows at once
    df = pd.DataFrame(all_rows, columns=mapping.keys())
    
    # Save final DataFrame to CSV
    df.to_csv('data.csv', index=False)
    print("Data saved to data.csv")

# Run the main function
if __name__ == "__main__":
    main()
