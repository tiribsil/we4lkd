import pandas as pd
import sys
from pathlib import Path
from utils import normalize_disease_name, get_logger

def link_abstracts(ground_truth_path, corpus_path, disease_name=None):
    logger = get_logger("LinkAbstracts")
    
    gt_path = Path(ground_truth_path)
    cp_path = Path(corpus_path)
    
    # Try to infer disease name if not provided
    if not disease_name:
        # Check if paths look like data/<disease>/...
        # Ground truth is typically in data/<disease>/ground_truth_cache/ground_truth_llm.csv
        parts = gt_path.parts
        if 'data' in parts:
            idx = parts.index('data')
            if idx + 1 < len(parts):
                disease_name = parts[idx+1]
                logger.info(f"Inferred disease name from path: {disease_name}")
        
    if not disease_name:
        logger.error("Disease name not provided and could not be inferred from paths.")
        print("Error: Disease name is required. Provide it as a 3rd argument or ensure files are in 'data/<disease>/' folders.")
        return

    # 1. Load data
    logger.info(f"Loading corpus from {corpus_path}")
    df_corpus = pd.read_csv(corpus_path)
    
    logger.info(f"Loading ground truth from {ground_truth_path}")
    df_gt = pd.read_csv(ground_truth_path)
    
    if df_corpus.empty or df_gt.empty:
        logger.error("One of the input files is empty.")
        return

    # 2. Process corpus exactly like extract_report_years.py
    norm_disease = normalize_disease_name(disease_name)
    logger.info(f"Filtering corpus for disease: {norm_disease}")
    
    # Ensure summary is string
    df_corpus['summary'] = df_corpus['summary'].astype(str)
    
    # Replicate the exact sorting and filtering from extract_report_years.py
    disease_mask = df_corpus['summary'].str.contains(norm_disease, case=False, na=False)
    disease_df = df_corpus[disease_mask].sort_values('year_extracted').copy()
    
    if disease_df.empty:
        logger.error(f"No abstracts found for disease '{norm_disease}' after filtering.")
        return

    # 3. Link based on abstract_id
    results = []
    
    # Check for ID columns
    possible_id_cols = ['pmid', 'id', 'pubmed_id', 'PMID', 'ID']
    found_id_col = None
    for col in possible_id_cols:
        if col in disease_df.columns:
            found_id_col = col
            break
            
    logger.info(f"Linking {len(df_gt)} compounds...")
    
    for _, gt_row in df_gt.iterrows():
        compound = gt_row['compound']
        abstract_id = gt_row['abstract_id']
        
        abstract_text = "NOT_FOUND"
        
        try:
            # The user said: "it's the line of a filtered cleaned abstracts"
            # This implies positional index (iloc) in the processed disease_df
            idx = int(float(abstract_id))
            if 0 <= idx < len(disease_df):
                abstract_text = disease_df.iloc[idx]['summary']
            else:
                # Fallback: maybe it's the original index label (loc)
                if abstract_id in disease_df.index:
                    abstract_text = disease_df.loc[abstract_id, 'summary']
                # Fallback: maybe it's an actual ID
                elif found_id_col:
                    match = disease_df[disease_df[found_id_col] == abstract_id]
                    if not match.empty:
                        abstract_text = match.iloc[0]['summary']
        except Exception as e:
            logger.warning(f"Error linking compound {compound} with ID {abstract_id}: {e}")
            
        results.append({
            'compound': compound,
            'abstract': abstract_text
        })
    
    # 4. Save results
    output_path = "linked_abstracts.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info(f"Saved linked abstracts to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python link_abstracts.py <ground_truth_csv> <clean_abstracts_csv> [disease_name]")
        sys.exit(1)
        
    gt_file = sys.argv[1]
    corpus_file = sys.argv[2]
    disease = sys.argv[3] if len(sys.argv) > 3 else None
    
    link_abstracts(gt_file, corpus_file, disease)
