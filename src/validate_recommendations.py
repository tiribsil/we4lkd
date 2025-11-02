import pandas as pd
import os
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from pathlib import Path
from utils import get_corpus_year_range, get_normalized_target_disease

def get_all_compounds(normalized_target_disease):
    """
    Reads the compound whitelist file and returns a list of all unique compound names
    that are also present in the final Word2Vec model's vocabulary.
    """
    compound_whitelist_file = f'data//compound_whitelist.txt'
    if not os.path.exists(compound_whitelist_file):
        print(f"Compound whitelist file not found: {compound_whitelist_file}")
        return []

    with open(compound_whitelist_file, 'r') as f:
        all_compounds = [line.strip() for line in f]

    # Filter compounds against the final model's vocabulary
    model_type = 'w2v' # Assuming w2v as per the request and existing code
    # The combination variable is no longer needed as we are directly targeting 'w2v_models'
    model_directory_path = Path(f'./data/{normalized_target_disease}/models/w2v_models/')

    # Find the latest model file by modification time
    model_files = sorted(
        model_directory_path.glob('*.model'),
        key=os.path.getmtime,
        reverse=True
    )

    if not model_files:
        print(f"No Word2Vec model files found in {model_directory_path}")
        return []

    final_model_path = model_files[0]
    print(f"Loading latest model from: {final_model_path}")

    final_model = Word2Vec.load(str(final_model_path))
    compounds_in_model_vocab = [compound for compound in all_compounds if compound in final_model.wv.key_to_index]
    print(f"Original compound list size: {len(all_compounds)}")
    print(f"Filtered compound list size (in model vocab): {len(compounds_in_model_vocab)}")
    return compounds_in_model_vocab
    
def get_year_reported(abstracts_file, target_disease, compounds, threshold=3):
    """
    Finds the first year a compound is mentioned with the target disease in an abstract,
    with the compound mentioned at least 'threshold' times.
    """
    if Path(abstracts_file).is_dir():
        # If it's a directory (Spark output), read all part-xxxx.csv files
        csv_files = list(Path(abstracts_file).glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in Spark output directory: {abstracts_file}")
            return {}
        
        list_df = []
        for f in csv_files:
            list_df.append(pd.read_csv(f))
        df = pd.concat(list_df, ignore_index=True)
    else:
        # If it's a single file (older output format or non-Spark)
        df = pd.read_csv(abstracts_file)
    year_reported = {}
    
    # Pre-compile regex for speed
    compound_regexes = {compound: re.compile(r'\b' + re.escape(compound) + r'\b', re.IGNORECASE) for compound in compounds}
    disease_regex = re.compile(r'\b' + re.escape(target_disease) + r'\b', re.IGNORECASE)

    for compound in tqdm(compounds, desc="Finding year reported"):
        # Find abstracts that contain both the compound and the disease
        compound_mentions_df = df[df['summary'].str.contains(compound_regexes[compound], na=False)]
        disease_mentions_df = compound_mentions_df[compound_mentions_df['summary'].str.contains(disease_regex, na=False)]
        
        if not disease_mentions_df.empty:
            # Count compound occurrences in the summary for the filtered abstracts
            compound_counts = disease_mentions_df['summary'].str.count(compound_regexes[compound])
            
            # Filter for abstracts where compound count meets the threshold
            eligible_mentions = disease_mentions_df[compound_counts >= threshold]

            if not eligible_mentions.empty:
                year_reported[compound] = eligible_mentions['year_extracted'].min()
            
    return year_reported

def get_first_recommendation(top_n_dir, compounds):
    """
    Finds the first year a compound is recommended in the top_n_score files.
    """
    first_recommendation = {}
    if not os.path.exists(top_n_dir):
        print(f"Directory not found: {top_n_dir}")
        return first_recommendation

    # Get all year directories and sort them to process chronologically
    try:
        year_dirs = sorted([d for d in os.listdir(top_n_dir) if os.path.isdir(os.path.join(top_n_dir, d)) and d.isdigit()], key=int)
    except Exception as e:
        print(f"Could not read year directories from {top_n_dir}: {e}")
        return first_recommendation

    # Create a set for faster lookups
    compounds_set = set(compounds)

    for year_str in tqdm(year_dirs, desc="Finding first recommendation"):
        year = int(year_str)
        year_path = os.path.join(top_n_dir, year_str)
        
        try:
            csv_files = [f for f in os.listdir(year_path) if f.endswith('.csv')]
        except Exception as e:
            print(f"Could not read files from {year_path}: {e}")
            continue

        for file in sorted(csv_files):
            file_path = os.path.join(year_path, file)
            try:
                df = pd.read_csv(file_path)
                
                column_name = None
                if 'compound_name' in df.columns:
                    column_name = 'compound_name'
                elif 'chemical_name' in df.columns:
                    column_name = 'chemical_name'

                if column_name:
                    # Iterate through compounds found in the file
                    for recommended_compound in df[column_name].values:
                        # If this compound is in our list and we haven't found a recommendation for it yet
                        if recommended_compound in compounds_set and recommended_compound not in first_recommendation:
                            first_recommendation[recommended_compound] = year
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    return first_recommendation

def main():
    """
    Main function to generate the compound table.
    """
    # Read target disease
    target_disease = get_normalized_target_disease()
    
    # Configuration
    abstracts_file = f'data/{target_disease}/corpus/clean_abstracts/clean_abstracts.csv'
    top_n_dir = f'data/{target_disease}/validation/w2v/top_n_compounds'
    model_type = 'w2v' # As per the existing code and directory structure
    output_file = f'data/{target_disease}/validation/{model_type}/recommendation_validation.csv'


    # Get all compounds
    compounds = get_all_compounds(target_disease)
    
    # Get year reported
    year_reported = get_year_reported(abstracts_file, target_disease, compounds, threshold=3)
    print(f"Found {len(year_reported)} compounds with a reported year.")
    
    # Get first recommendation
    first_recommendation = get_first_recommendation(top_n_dir, compounds)
    print(f"Found {len(first_recommendation)} compounds with a first recommendation year.")
    
    # Create the table
    table_data = []
    for compound in compounds:
        report_year = year_reported.get(compound)
        recommend_year = first_recommendation.get(compound)
        
        if report_year and recommend_year:
            how_early = report_year - recommend_year
            table_data.append([compound, report_year, recommend_year, how_early])
    
    print(f"Found {len(table_data)} compounds with both a reported year and a recommendation year.")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_data, columns=['compound_name', 'year_reported', 'first_recommendation', 'how_early'])
    
    # Ensure the output directory exists
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Table saved to {output_file}")

if __name__ == '__main__':
    main()
