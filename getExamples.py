import pandas as pd
import requests
import io

def create_tess_csv():
    # The direct CSV export link for the TESS Objects of Interest catalog
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    
    # Headers to prevent the server from blocking the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Connecting to NASA/ExoFOP servers...")
    
    try:
        # Fetch the data
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Load into pandas
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean column names (lowercase and no spaces)
        df.columns = df.columns.str.strip().str.lower()
        
        # To make sure that our entries are most likely in the TESS TPF 2-minute cadence library 
        df = df[df['tess mag'] < 13] # Looks for brighter stars 
        
        # Filtering for the three specific classes
        # Note | We are classifying based on the 'tfopwg disposition' column 
        positives = df[df['tfopwg disposition'].isin(['KP', 'CP'])].head(500)
        negatives = df[df['tfopwg disposition'] == 'FP'].head(500)
        
        # 3. Combine them
        relevant_columns = ['tic id', 'toi', 'tfopwg disposition', 'sectors']
        final_data = pd.concat([positives, negatives])[relevant_columns]
        
        # 4. Save to CSV
        final_data.to_csv("classified_targets.csv", index=False)
        
        print(f"Success! Created 'classified_targets.csv' with {len(final_data)} examples in total.")
        print(f"Contains: {len(positives)} Positive Targets (KP/CP), {len(negatives)} Negative Targets (FP)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_tess_csv()