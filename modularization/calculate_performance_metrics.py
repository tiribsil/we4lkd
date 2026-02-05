import argparse
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Tuple

# Assuming utils is in the same directory, or standardized project structure
try:
    from utils import get_logger
except ImportError:
    # Fallback if running standalone and path issues
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = get_logger("PerformanceCalculator")

def load_ground_truth(file_path: Path) -> Dict[str, int]:
    """
    Loads the ground truth from a CSV file.
    Expected columns: 'compound' (or 'chemical_name') and 'year' (or 'first_report_year').
    """
    if not file_path.exists():
        logger.error(f"Ground truth file not found: {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    # Normalize column names
    cols = df.columns.tolist()
    compound_col = None
    year_col = None
    
    possible_compound_cols = ['compound', 'chemical_name', 'compound_name', 'name']
    possible_year_cols = ['year', 'report_year', 'first_report_year', 'year_extracted']
    
    for c in possible_compound_cols:
        if c in cols:
            compound_col = c
            break
            
    for c in possible_year_cols:
        if c in cols:
            year_col = c
            break
            
    if not compound_col or not year_col:
        logger.error(f"Could not identify compound or year columns in {file_path}. Found: {cols}")
        sys.exit(1)
        
    return dict(zip(df[compound_col], df[year_col]))

def calculate_metrics(
    ground_truth: Dict[str, int],
    base_path: Path,
    start_year: int,
    end_year: int,
    output_path: Path
):
    """
    Calculates performance metrics based on the intersection of recommendations and ground truth.
    Miminics logic from model_evaluation.py
    """
    first_recommendation = {}
    
    logger.info(f"Processing years {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        year_path = base_path / str(year)
        if not year_path.exists():
            logger.warning(f"Directory not found for year {year}: {year_path}")
            continue
            
        csv_files = list(year_path.glob("top_*_score.csv"))
        if not csv_files:
            logger.warning(f"No top scores file found in {year_path}")
            continue
        
        # Use simple heuristic: take the first one found or sort? 
        # Original code: csv_files[0]
        target_file = csv_files[0]
        
        try:
            df = pd.read_csv(target_file)
            col = 'chemical_name' if 'chemical_name' in df.columns else 'compound_name' # fallback
            if 'compound' in df.columns: col = 'compound' 
            
            if col in df.columns:
                for compound in df[col].values:
                    # Logic: record the FIRST year we recommended it
                    if compound not in first_recommendation and compound in ground_truth:
                        first_recommendation[compound] = year
            else:
                logger.warning(f"Column for compound name not found in {target_file}")
                
        except Exception as e:
            logger.error(f"Error reading {target_file}: {e}")

    # Prepare details
    details = []
    skipped_count = 0
    
    for compound, rec_year in first_recommendation.items():
        report_year = ground_truth[compound]
        
        # Mimicking the logic in model_evaluation.py:
        # "if report_year < self.test_start_year: continue"
        if report_year < start_year:
            skipped_count += 1
            continue
            
        how_early = report_year - rec_year
        details.append((compound, rec_year, report_year, how_early))

    logger.info(f"Skipped {skipped_count} matches where report_year < start_year.")

    if not details:
        logger.warning("No intersection found between recommendations and ground truth in the valid test period.")
        return

    details_df = pd.DataFrame(details, columns=['compound', 'recommendation_year', 'literature_report_year', 'years_early'])
    
    # Statistics
    mean_early = details_df['years_early'].mean()
    median_early = details_df['years_early'].median()
    std_dev = details_df['years_early'].std() if len(details_df) > 1 else 0.0
    mode = details_df['years_early'].mode().tolist()
    
    # Print Metrics
    print(f"\n{'='*40}")
    print(f"Final Performance Metrics ({start_year}-{end_year})")
    print(f"{'='*40}")
    print(f"Total Valid Matches: {len(details_df)}")
    print(f"Anticipation Mean:   {mean_early:.2f}")
    print(f"Anticipation Median: {median_early}")
    print(f"Anticipation Std Dev:{std_dev:.2f}")
    print(f"Anticipation Mode:   {mode}")
    
    print("\nTop 5 Biggest Anticipations (Found Early):")
    top_5_biggest = details_df.nlargest(5, 'years_early')
    for _, row in top_5_biggest.iterrows():
        print(f"  {row['years_early']} years - {row['compound']} (Rec: {row['recommendation_year']}, Report: {row['literature_report_year']})")

    print("\nTop 5 Smallest (Delays/Late):")
    top_5_smallest = details_df.nsmallest(5, 'years_early')
    for _, row in top_5_smallest.iterrows():
        print(f"  {row['years_early']} years - {row['compound']} (Rec: {row['recommendation_year']}, Report: {row['literature_report_year']})")
    print(f"{'='*40}\n")
    
    # Save CSV
    try:
        if output_path.is_dir():
            output_path = output_path / "final_performance.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(output_path, index=False)
        logger.info(f"Full details saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate final performance metrics for model recommendations.")
    
    parser.add_argument("--ground-truth", "-g", required=True, type=Path, help="Path to ground truth CSV file")
    parser.add_argument("--top-n-path", "-p", required=True, type=Path, help="Base path containing year folders with top_n_scores")
    parser.add_argument("--start-year", "-s", required=True, type=int, help="Start year of the test period")
    parser.add_argument("--end-year", "-e", required=True, type=int, help="End year of the test period")
    parser.add_argument("--output", "-o", type=Path, default=Path("final_performance_metrics.csv"), help="Output CSV file path")
    
    args = parser.parse_args()
    
    logger.info("Initializing Performance Calculator...")
    logger.info(f"Ground Truth: {args.ground_truth}")
    logger.info(f"Top N Path:   {args.top_n_path}")
    logger.info(f"Period:       {args.start_year}-{args.end_year}")
    
    gt_dict = load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(gt_dict)} items from ground truth.")
    
    calculate_metrics(gt_dict, args.top_n_path, args.start_year, args.end_year, args.output)

if __name__ == "__main__":
    main()
