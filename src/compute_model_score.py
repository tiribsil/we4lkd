import pandas as pd
from pathlib import Path

from utils import get_normalized_target_disease


def main():
    target_disease = get_normalized_target_disease()
    
    model_type = 'w2v'
    output_file = f'data/{target_disease}/validation/{model_type}/recommendation_validation.csv'

    df = pd.read_csv(output_file)
    
    print(f"The model recommended compounds {df['how_early'].mean():.2f} years before their reports, on average.")

    # biggest (and what compound was it), smallest (and what compound), median, standard deviation, mode
    
    top_5_biggest = df.nlargest(5, 'how_early')
    top_5_smallest = df.nsmallest(5, 'how_early')
    median = df['how_early'].median()
    std_dev = df['how_early'].std()
    mode = df['how_early'].mode().to_list()
    
    print("\nTop 5 Biggest Differences:")
    for index, row in top_5_biggest.iterrows():
        print(f"  {row['how_early']} years for compound '{row['compound_name']}'")

    print("\nTop 5 Smallest Differences:")
    for index, row in top_5_smallest.iterrows():
        print(f"  {row['how_early']} years for compound '{row['compound_name']}'")

    print(f"\nMedian: {median} years")
    print(f"Standard Deviation: {std_dev:.2f} years")
    print(f"Mode(s): {mode} years")


if __name__ == '__main__':
    main()
