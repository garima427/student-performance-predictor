"""
Model Training for Student Performance Predictor.
Trains: Simple Linear Regression, Multiple Linear Regression, KNN Regressor.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append(str(Path(__file__).parent))
from data_preprocessing import (
    FEATURES, TARGET, load_data, clean_data, prepare_features,
    split_data, scale_features
)

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate regression model with standard metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"model": model_name, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "model_obj": model}


def train_simple_linear(X_train, y_train, X_test, y_test):
    """
    Simple Linear Regression using only the most correlated feature.
    We pick previous_score as it typically correlates most with final score.
    """
    # Use only previous_score for simple regression
    feature_idx = FEATURES.index("previous_score")
    X_train_simple = X_train.iloc[:, [feature_idx]] if hasattr(X_train, 'iloc') else X_train[:, [feature_idx]]
    X_test_simple = X_test.iloc[:, [feature_idx]] if hasattr(X_test, 'iloc') else X_test[:, [feature_idx]]
    
    model = LinearRegression()
    model.fit(X_train_simple, y_train)
    print(f"Simple Linear Regression: predicting final_score from previous_score")
    print(f"  Coefficient: {model.coef_[0]:.3f}, Intercept: {model.intercept_:.3f}")
    return model, evaluate_model(model, X_test_simple, y_test, "Simple Linear Regression"), X_test_simple


def train_multiple_linear(X_train, y_train, X_test, y_test):
    """Multiple Linear Regression using all features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Explain coefficients
    print("\nMultiple Linear Regression coefficients:")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat}: {coef:.3f}")
    print(f"  Intercept: {model.intercept_:.3f}")
    
    return model, evaluate_model(model, X_test, y_test, "Multiple Linear Regression")


def train_knn(X_train, y_train, X_test, y_test):
    """KNN Regressor with different K values comparison."""
    print("\n=== KNN K-Value Experimentation ===")
    best_k, best_rmse = None, float('inf')
    results = []
    
    for k in [3, 5, 7, 9, 11]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((k, rmse))
        print(f"  K={k}: RMSE={rmse:.2f}")
        if rmse < best_rmse:
            best_rmse, best_k = rmse, k
    
    print(f"Best K: {best_k} (RMSE: {best_rmse:.2f})")
    
    # Train final model with best K
    model = KNeighborsRegressor(n_neighbors=best_k)
    model.fit(X_train, y_train)
    
    # Plot K comparison
    plt.figure(figsize=(8, 5))
    ks = [r[0] for r in results]
    rmses = [r[1] for r in results]
    plt.plot(ks, rmses, 'o-')
    plt.xlabel("K Value")
    plt.ylabel("RMSE")
    plt.title("KNN: K Value vs RMSE")
    plt.savefig("eda_knn_k_values.png")
    plt.close()
    
    return model, evaluate_model(model, X_test, y_test, f"KNN Regressor (K={best_k})")


def plot_predictions(y_test, predictions_dict):
    """Plot actual vs predicted for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("predictions_comparison.png")
    plt.close()
    print("Saved: predictions_comparison.png")


def main():
    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess
    df = load_data()
    df = clean_data(df)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    
    # Train all three models
    simple_model, simple_results, X_test_simple = train_simple_linear(X_train_s, y_train, X_test_s, y_test)
    multi_model, multi_results = train_multiple_linear(X_train_s, y_train, X_test_s, y_test)
    knn_model, knn_results = train_knn(X_train_s, y_train, X_test_s, y_test)
    
    # Comparison table
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'model_obj'}
        for r in [simple_results, multi_results, knn_results]
    ])
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Select best by R²
    best_idx = results_df['r2'].idxmax()
    best_name = results_df.loc[best_idx, 'model']
    print(f"\nBest model: {best_name} (R² = {results_df.loc[best_idx, 'r2']:.4f})")
    
    # Save models and scaler
    joblib.dump(simple_model, MODELS_DIR / "simple_linear.joblib")
    joblib.dump(multi_model, MODELS_DIR / "multiple_linear.joblib")
    joblib.dump(knn_model, MODELS_DIR / "knn.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"\nModels saved to {MODELS_DIR}/")
    
    # Plot predictions
    preds = {
        "Simple Linear": simple_model.predict(X_test_simple),
        "Multiple Linear": multi_model.predict(X_test_s),
        "KNN": knn_model.predict(X_test_s)
    }
    plot_predictions(y_test, preds)


if __name__ == "__main__":
    main()