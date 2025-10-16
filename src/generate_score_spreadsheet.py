import pandas as pd
import os

from utils import *

def create_compound_spreadsheet(data_path, output_path):
    """
    Creates a spreadsheet of top scoring compounds per year.

    Args:
        data_path (str): The path to the directory containing the year folders.
        output_path (str): The path to save the output CSV file.
    """
    year_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    year_dirs.sort()

    all_compounds = {}
    for year in year_dirs:
        file_path = os.path.join(data_path, year, 'top_20_score.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_compounds[year] = (df['chemical_name'] + ' (' + df['score'].map('{:.4f}'.format) + ')').tolist()
            except pd.errors.EmptyDataError:
                print(f"Warning: {file_path} is empty and will be skipped.")
                continue

    max_compounds = max(len(v) for v in all_compounds.values())
    
    for year in all_compounds:
        if len(all_compounds[year]) < max_compounds:
            all_compounds[year].extend([None] * (max_compounds - len(all_compounds[year])))

    df_final = pd.DataFrame(all_compounds)
    df_final.to_csv(output_path, index=False)
    print(f"Spreadsheet saved to {output_path}")

if __name__ == '__main__':
    normalized_target_disease = get_normalized_target_disease()
    data_path = f'./data/{normalized_target_disease}/validation/w2v/top_n_compounds/'
    output_path = f'./data/{normalized_target_disease}/top_compounds_by_year.csv'
    create_compound_spreadsheet(data_path, output_path)
