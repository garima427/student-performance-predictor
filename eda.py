"""
Exploratory Data Analysis for Student Performance Predictor.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_preprocessing import FEATURES, TARGET, load_data, clean_data


def run_eda(df):
    """Perform EDA with visualizations."""
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # 1. Distribution of final scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=20, kde=True)
    plt.title("Distribution of Final Exam Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Count")
    plt.savefig("eda_final_score.png")
    plt.close()
    print("Saved: eda_final_score.png")
    
    # 2. Study hours vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="study_hours", y=TARGET, data=df)
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Final Score")
    plt.savefig("eda_study_hours.png")
    plt.close()
    print("Saved: eda_study_hours.png")
    
    # 3. Attendance vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="attendance", y=TARGET, data=df)
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    plt.savefig("eda_attendance.png")
    plt.close()
    print("Saved: eda_attendance.png")
    
    # 4. Previous score vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="previous_score", y=TARGET, data=df)
    plt.title("Previous Score vs Final Score")
    plt.xlabel("Previous Exam Score")
    plt.ylabel("Final Score")
    plt.savefig("eda_previous_score.png")
    plt.close()
    print("Saved: eda_previous_score.png")
    
    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("eda_correlation.png")
    plt.close()
    print("Saved: eda_correlation.png")
    
    # Print key correlations
    print("\nTop correlations with final_score:")
    corr_with_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    for feature, corr_val in corr_with_target.items():
        print(f"  {feature}: {corr_val:.3f}")


def main():
    df = load_data()
    df = clean_data(df)
    run_eda(df)


if __name__ == "__main__":
    main()