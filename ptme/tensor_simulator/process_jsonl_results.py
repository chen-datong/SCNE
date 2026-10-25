import json
import pandas as pd
import os
import glob
from pathlib import Path

def process_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filepath}: {e}")
                continue
    
    if not data:
        return []

    processed = []
    for entry in data:
        if entry.get('MaxDim') != 800:
            continue
        # Retrieve moments calculated in C++
        # If keys are missing, default to 0.0 or handle appropriately
        M2 = entry.get('M2', 0.0)
        M3 = entry.get('M3', 0.0)
        M4 = entry.get('M4', 0.0)
        M5 = entry.get('M5', 0.0)
        
        # Apply formulas from negativity_estimation.py
        
        purity = M2 - 1.0
        thirdM = (M3 - 1.0 - 3.0 * purity) / 2.0
        fourthM = (M4 - 1.0 - 6.0 * purity - 8.0 * thirdM - 3.0 * (purity ** 2)) / 6.0
        fifthM = (M5 - 1.0 - 10.0 * purity - 20.0 * thirdM - 15.0 * (purity ** 2)
                  - 30.0 * fourthM - 20.0 * purity * thirdM) / 24.0
        
        row = {
            'nA': entry.get('N')-2,
            'nB': 2,
            'Nu': 1,
            'NM': entry.get('nSamples'),
            'dprob': entry.get('p'),
            # Processed values
            'purity': purity,
            'thirdM': thirdM,
            'fourthM': fourthM,
            'fifthM': fifthM
        }
        processed.append(row)
    return processed

def main():
    # Directory assumed from main.cpp: "TIM_complete"
    data_root = Path(__file__).resolve().parents[2] / 'data' / 'ptme'
    input_dir = data_root / 'tensor_simulator' / 'TIM_complete'
    output_file = data_root / 'processed_results2.csv'
    
    if not os.path.exists(input_dir):
        # Create it if it doesn't exist so the script doesn't just fail immediately if run empty
        # But here we probably want to warn.
        print(f"Directory '{input_dir}' does not exist. Please check your C++ output directory.")
        return

    all_data = []
    files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    
    print(f"Found {len(files)} .jsonl files in {input_dir}")
    
    for filepath in files:
        file_data = process_file(filepath)
        all_data.extend(file_data)
        
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Sort columns for nicer output
        cols = ['nA', 'nB', 'Nu', 'NM', 'dprob', 'purity', 'thirdM', 'fourthM', 'fifthM']
        # Reorder if columns exist
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        # Sort rows
        sort_cols = [c for c in ['nA', 'dprob', 'NM'] if c in df.columns]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df)} records.")
        print(f"Results saved to {output_file}")
        
        # Display first few rows
        print(df.head())
    else:
        print("No valid data found in the JSONL files.")

if __name__ == "__main__":
    main()
