"""
Data Preprocessing for Student Performance Predictor.
Handles loading, cleaning, encoding, scaling and train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configuration
RANDOM_STATE = 42
TARGET = "final_score"

# Feature definitions
FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
    "extracurricular"
]


def load_data(filepath="data/students_real.csv"):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def inspect_data(df):
    """Inspect dataset for quality issues."""
    print("\n=== DATA INSPECTION ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print("\nBasic statistics:")
    print(df.describe().round(2))
    print("\nClass/Target distribution:")
    print(df[TARGET].describe())


def clean_data(df):
    """Clean data: remove duplicates, handle missing values."""
    print("\n=== DATA CLEANING ===")
    initial_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
    
    # Handle missing values with median imputation
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in {col} with median: {median_val:.2f}")
    
    # Ensure numerical types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Final shape: {df.shape}")
    return df


def encode_categorical(df):
    """Encode categorical variables (extracurricular: 0/1 already binary)."""
    print("\n=== FEATURE ENCODING ===")
    # extracurricular is already binary (0/1), no encoding needed
    print("extracurricular is already binary (0=No, 1=Yes)")
    return df


def prepare_features(df):
    """Separate features (X) and target (y)."""
    X = df[FEATURES]
    y = df[TARGET]
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {FEATURES}")
    return X, y


def split_data(X, y, test_size=0.2):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"\n=== TRAIN/TEST SPLIT ===")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def create_scaler():
    """Create a StandardScaler for feature normalization."""
    return StandardScaler()


def scale_features(X_train, X_test):
    """Scale features using StandardScaler (fit on train only)."""
    print("\n=== FEATURE SCALING ===")
    scaler = create_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled training data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


def main():
    """Main preprocessing pipeline."""
    print("=" * 50)
    print("STUDENT PERFORMANCE PREDICTOR - PREPROCESSING")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Inspect
    inspect_data(df)
    
    # Clean
    df_clean = clean_data(df)
    
    # Encode
    df_encoded = encode_categorical(df_clean)
    
    # Prepare features
    X, y = prepare_features(df_encoded)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale (important for KNN)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()