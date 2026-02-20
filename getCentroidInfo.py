import pandas as pd
import lightkurve as lk
import numpy as np
import os
import shutil
import time 
import random 
from tqdm import tqdm  # The progress bar library

def process_targets(manifest_path="classified_targets.csv", output_path="tess_training_data.csv"):
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Ensure that you have run getExamples.py to create the file classified_targets.csv first.")
        return
    
    manifest = pd.read_csv(manifest_path)
    final_data = []
    cache_dir = "./tpf_temp"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Initializing TESS Data Extraction ipeline...P")
    
    # tqdm will automatically calculate the time remaining based on the length of the manifest
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing Targets", unit="star"):
        tic_id = f"TIC {row['tic id']}"
        label = row['tfopwg disposition']
        target_sector = int(str(row['sectors']).split(',')[0]) 
        
        try:
            # This random sleep is to be polite to the servers and avoid hitting rate limits, especially if we have many targets. 
            # If this isn't here, it hangs after 200 targets because of rate limiting.
            time.sleep(random.uniform(0.5, 2.0))
            # By searching for a pre-processed lightcurve, we can avoid downloading large TPF files and performing photometry locally, which is much faster 
            search = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC", cadence="short")
            
            if len(search) > 0:
                # Download the lightcurve file, which is much smaller and faster.
                lc = search[-1].download(download_dir=cache_dir, cutout_size=11, quality_bitmask="default")

                # If download fails or the file is corrupt, lc can be None.
                if lc is None:
                    continue
                
                # Standardizing it to 1000 data points keeps some ML consistency, padding with NaN if shorter 
                length = 1000
                
                flux_orig = lc.flux.value
                c_col_orig = lc.centroid_col.value
                c_row_orig = lc.centroid_row.value

                flux = np.full(length, np.nan)
                c_col = np.full(length, np.nan)
                c_row = np.full(length, np.nan)

                actual_len = min(len(flux_orig), length)
                flux[:actual_len] = flux_orig[:actual_len]
                c_col[:actual_len] = c_col_orig[:actual_len]
                c_row[:actual_len] = c_row_orig[:actual_len]

                entry = {'tic_id': row['tic id'], 'label': label}
                
                # Dynamic column creation for Flux and Centroids
                for i in range(len(flux)):
                    entry[f'f_{i}'] = flux[i]
                    entry[f'cc_{i}'] = c_col[i]
                    entry[f'cr_{i}'] = c_row[i]
                
                final_data.append(entry)
                
        except Exception:
            # We skip errors silently to keep the progress bar clean
            continue

    # Convert to CSV
    if final_data:
        pd.DataFrame(final_data).to_csv(output_path, index=False)
        print(f"\nSaved {len(final_data)} targets to {output_path}")
    
    # Cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Temporary TPF files removed.")

if __name__ == "__main__":
    process_targets()