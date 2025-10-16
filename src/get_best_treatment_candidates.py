import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import pandas as pd
import sys

from src.utils import *

TOP_N = 10
HEADER_NAME = 'chemical_name'

def main():
    """
    Reads the top compounds from the latest year, based on the 'score' metric,
    and saves them as potential treatment candidates.
    """
    normalized_target_disease = get_normalized_target_disease()
    _, end_year = get_corpus_year_range(normalized_target_disease)
    model_type = 'w2v'
    top_compounds_path = Path(f'./data/{normalized_target_disease}/validation/{model_type}/top_n_compounds')

    latest_year_dir = top_compounds_path / str(end_year)
    score_files = list(latest_year_dir.glob('*score.csv'))

    if not score_files:
        print(f"Score file not found in '{latest_year_dir.resolve()}'.", file=sys.stderr)
        print("Please ensure the latent knowledge report has been generated with the 'score' metric.", file=sys.stderr)
        return

    score_file = score_files[0]

    try:
        df = pd.read_csv(score_file)
        potential_treatments = df[HEADER_NAME].head(TOP_N).tolist()
    except Exception as e:
        print(f"Error reading score file: {e}", file=sys.stderr)
        return

    output_dir = Path(f'./data/{normalized_target_disease}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'potential_treatments.txt'

    with open(output_path, 'w', encoding='utf-8') as f:
        for treatment in potential_treatments:
            f.write(treatment + '\n')

    print(f"Top {len(potential_treatments)} potential treatments from {end_year} saved to {output_path}")

    print(f"\n{'Compound'}")
    print("-" * 60)
    for treatment in potential_treatments:
        print(treatment)


if __name__ == "__main__":
    main()
