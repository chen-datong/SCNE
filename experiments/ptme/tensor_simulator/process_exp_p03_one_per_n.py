import glob
import json
import os
import re

import pandas as pd


EXP_DIR = "exp"
OUTPUT_FILE = "processed_results_p0.30_one_per_n.csv"


def parse_sample_count(filepath: str) -> int:
    filename = os.path.basename(filepath)
    match = re.search(r"_Sample(\d+)\.jsonl$", filename)
    return int(match.group(1)) if match else -1


def parse_n_from_filename(filepath: str) -> int:
    filename = os.path.basename(filepath)
    match = re.search(r"^n(\d+)_", filename)
    return int(match.group(1)) if match else -1


def select_one_file_per_n():
    pattern = os.path.join(EXP_DIR, "n*_depth*_J*_h*_p0.30_Sample*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return []

    selected = {}
    for filepath in files:
        n_value = parse_n_from_filename(filepath)
        if n_value < 0:
            continue
        sample_count = parse_sample_count(filepath)
        prev = selected.get(n_value)
        if prev is None or sample_count > prev[0]:
            selected[n_value] = (sample_count, filepath)

    return [selected[n][1] for n in sorted(selected.keys())]


def process_file(filepath: str):
    processed = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filepath}: {e}")
                continue

            if entry.get("MaxDim") != 800:
                continue

            m2 = entry.get("M2", 0.0)
            m3 = entry.get("M3", 0.0)
            m4 = entry.get("M4", 0.0)
            m5 = entry.get("M5", 0.0)

            purity = m2 - 1.0
            third_m = (m3 - 1.0 - 3.0 * purity) / 2.0
            fourth_m = (m4 - 1.0 - 6.0 * purity - 8.0 * third_m - 3.0 * (purity ** 2)) / 6.0
            fifth_m = (
                m5
                - 1.0
                - 10.0 * purity
                - 20.0 * third_m
                - 15.0 * (purity ** 2)
                - 30.0 * fourth_m
                - 20.0 * purity * third_m
            ) / 24.0

            row = {
                "source_file": os.path.basename(filepath),
                "nA": entry.get("N") - 2 if entry.get("N") is not None else None,
                "nB": 2,
                "Nu": 1,
                "NM": entry.get("nSamples"),
                "dprob": entry.get("p"),
                "purity": purity,
                "thirdM": third_m,
                "fourthM": fourth_m,
                "fifthM": fifth_m,
            }
            processed.append(row)
    return processed


def main():
    if not os.path.exists(EXP_DIR):
        print(f"Directory '{EXP_DIR}' does not exist.")
        return

    selected_files = select_one_file_per_n()
    print(f"Selected {len(selected_files)} files (one per N) for p=0.30:")
    for filepath in selected_files:
        print(f"  - {os.path.basename(filepath)}")

    all_data = []
    for filepath in selected_files:
        all_data.extend(process_file(filepath))

    if not all_data:
        print("No valid data found.")
        return

    df = pd.DataFrame(all_data)
    cols = ["source_file", "nA", "nB", "Nu", "NM", "dprob", "purity", "thirdM", "fourthM", "fifthM"]
    df = df[[c for c in cols if c in df.columns]]
    sort_cols = [c for c in ["nA", "NM"] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Successfully processed {len(df)} records.")
    print(f"Results saved to {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
