import os
import sys
from pathlib import Path

from utils import *

TOP_N = 30
HEADER_NAME = 'chemical_name'

os.chdir(Path(__file__).resolve().parent.parent)


def main():
    """
    Reads compound data from year-based subdirectories, calculates a score
    based on occurrence count and consistency over time, and prints the top results.
    A compound is only counted once per year, regardless of how many files it appears in.
    """
    normalized_target_disease = get_normalized_target_disease()
    _, end_year = get_corpus_year_range(normalized_target_disease)

    # Data structure tracks total count and the set of years a compound appeared in.
    # {'compound': {'count': int, 'years_mentioned': set()}}
    compound_data = {}
    model_type = 'w2v'
    top_compounds_path = Path(f'./data/{normalized_target_disease}/validation/{model_type}/top_n_compounds')

    year_dirs = sorted(
        [p for p in top_compounds_path.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

    if not year_dirs:
        print(f"No year-named directories found in '{top_compounds_path.resolve()}'.", file=sys.stderr)
        return

    for year_dir in year_dirs:
        current_year = int(year_dir.name)
        compounds_this_year = set()

        for file_path in year_dir.glob('*'):
            if not file_path.is_file():
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    compound = line.strip()
                    if compound and compound != HEADER_NAME:
                        compounds_this_year.add(compound)

        # After processing all files for the year, update the master data.
        # This ensures each compound is counted only once for the current year.
        for compound in compounds_this_year:
            if compound not in compound_data:
                compound_data[compound] = {
                    'count': 0,
                    'years_mentioned': set()
                }
            compound_data[compound]['count'] += 1
            compound_data[compound]['years_mentioned'].add(current_year)

    scored_results = []
    for compound, data in compound_data.items():
        # Note: `count` now equals `len(years_mentioned)` due to the processing logic above.
        count = data['count']
        years = data['years_mentioned']

        if not years:
            continue

        first_year = min(years)
        unique_years_count = len(years)

        time_span = end_year - first_year + 1

        consistency_ratio = unique_years_count / time_span
        score = count * consistency_ratio

        scored_results.append({
            'name': compound,
            'score': score,
            'first_year': first_year,
            'count': count # This is the count of unique years.
        })

    scored_results.sort(key=lambda x: x['score'], reverse=True)

    # --- Gemini Edit Start ---
    # Save the top N chemical names to a file
    output_dir = Path(f'./data/{normalized_target_disease}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'potential_treatments.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in scored_results[:TOP_N]:
            f.write(result['name'] + '\n')
    print(f"\nTop {TOP_N} potential treatments saved to {output_path}")
    # --- Gemini Edit End ---

    print(f"\n{'Score':<12} {'Compound'}")
    print("-" * 60)
    for result in scored_results[:TOP_N]:
        print(
            f"{result['score']:<12.4f} {result['name']} "
            f"({result['count']} years, first seen {end_year - result['first_year'] + 1} years ago ({result['first_year']}))"
        )


if __name__ == "__main__":
    main()
