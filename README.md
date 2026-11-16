# Student Performance Predictor

A beginner-friendly machine learning project that predicts a student's final exam score (0–100) based on academic and lifestyle factors.

## Problem Statement

Predict a student's final exam score using features like study hours, attendance, previous scores, sleep, assignment completion, and extracurricular activities.

## Dataset

**Source:** UCI Machine Learning Repository — Student Performance Data Set
- **URL:** https://archive.ics.uci.edu/dataset/320/student+performance
- **File:** `student-mat.csv` (Portuguese secondary school, Mathematics course)
- **Records:** 395 students × 33 original features → mapped to 7 project features

**Feature Mapping (UCI → Project):**
| UCI Feature | Project Feature | Mapping |
|-------------|----------------|---------|
| `studytime` (1–4 weekly hours) | `study_hours` | Converted to daily hours |
| `absences` | `attendance` | Inverted: 100 − absences×1.5 |
| `G1`, `G2` (grades 0–20) | `previous_score` | Average scaled to 0–100 |
| `health`, `freetime` | `sleep_hours` | Estimated proxy (5–9 hrs) |
| `schoolsup`, `famsup`, `paid` | `assignment_completion` | Composite score 20–100% |
| `activities` (yes/no) | `extracurricular` | Binary 0/1 |
| `G3` (final grade 0–20) | `final_score` | Scaled to 0–100 |

**Note:** `prepare_real_data.py` contains the full mapping logic if you want to reproduce or modify it.

## Models Used

1. **Simple Linear Regression** — Uses only `previous_score` to predict final score (baseline)
2. **Multiple Linear Regression** — Uses all 6 features
3. **KNN Regressor** — Predicts based on similar students (requires feature scaling)

## Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute prediction error
- **MSE** (Mean Squared Error): Average squared error
- **RMSE** (Root Mean Squared Error): MSE in original units
- **R²** (R-squared): Proportion of variance explained (higher = better)

## Project Structure

```
student-performance-predictor/
├── data/
│   └── students.csv           # Dataset
├── data_preprocessing.py      # Data cleaning & preprocessing
├── eda.py                     # Exploratory Data Analysis
├── model_training.py          # Model training & evaluation
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python data_preprocessing.py
```

### 3. Run EDA
```bash
python eda.py
```

### 4. Train Models
```bash
python model_training.py
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

## Results (Real UCI Dataset — 395 students)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Simple Linear Regression** ✅ Best | 7.55 | 10.84 | 0.771 |
| Multiple Linear Regression | 8.08 | 11.24 | 0.754 |
| KNN Regressor (K=7) | 9.23 | 13.40 | 0.650 |

**Key insight:** On real data, `previous_score` alone is highly predictive (correlation 0.89), so Simple Linear Regression outperforms more complex models — a great lesson in Occam's razor.

## Limitations

- Synthetic dataset (not real student data)
- Small dataset size (~40 samples)
- Simple linear relationships assumed
- No temporal/seasonal factors

## Future Improvements

- Use real student performance dataset
- Add more features (parental education, distance to school)
- Try more algorithms (Random Forest, XGBoost)
- Add hyperparameter tuning
- Deploy to cloud