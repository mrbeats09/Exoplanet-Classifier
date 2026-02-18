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
        
        # 1. Clean column names (lowercase and no spaces)
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. Filter for our three specific classes
        # 'tfopwg disposition' is the standard column name in this CSV
        positives = df[df['tfopwg disposition'].isin(['KP', 'CP'])].head(250)
        negatives = df[df['tfopwg disposition'] == 'FP'].head(250)
        
        # 3. Combine them
        final_500 = pd.concat([positives, negatives])
        
        # 4. Save to CSV
        final_500.to_csv("classified_targets.csv", index=False)
        
        print(f"Success! Created 'classified_targets.csv' with {len(final_500)} rows.")
        print(f"Breakdown: {len(positives)} Positive (KP/CP), {len(negatives)} Negative (FP)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_tess_csv()