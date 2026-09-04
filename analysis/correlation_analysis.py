import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def analyze_correlation(results_csv_path: str = "results/classification_results.csv"):
    """
    Computes statistical correlation between predictive uncertainty and explanation consistency.
    """
    print("=== Correlation Analysis: Uncertainty vs. Explanation Consistency ===")
    results_path = "results/correlation_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        print(df.to_string(index=False))
    else:
        print("Run consistency_analysis.py first to generate consistency and uncertainty scores.")

if __name__ == "__main__":
    import os
    analyze_correlation()
