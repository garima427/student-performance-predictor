"""
Prepare the real UCI Student Performance dataset for the project.
Maps UCI features to the features used by the project.

UCI dataset columns (semicolon-separated):
school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,
guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,
nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,
absences,G1,G2,G3
"""

import pandas as pd
import numpy as np

# Load the larger dataset (student-mat.csv, 395 students)
df = pd.read_csv("data/student-mat.csv", sep=";")
print(f"Original UCI dataset: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Map UCI features to project features
# UCI studytime: 1-4 (weekly hours: <2, 2-5, 5-10, >10)
# Project study_hours: daily hours 0.5-10 -> approximate daily from weekly
studytime_map = {1: 1.5, 2: 4.0, 3: 7.0, 4: 10.0}  # weekly -> approx daily avg
df["study_hours"] = df["studytime"].map(studytime_map) * 7 / 5  # rough daily equivalent

# Attendance: UCI has absences -> invert to attendance %
df["attendance"] = np.clip(100 - (df["absences"] * 1.5), 40, 100)

# Previous score: use G1 or G2 (0-20 scale) -> scale to 0-100
df["previous_score"] = ((df["G1"] + df["G2"]) / 2) * 5  # avg of G1,G2 scaled to 0-100

# Sleep hours: not in UCI -> estimate from health/freetime (reasonable proxy)
df["sleep_hours"] = 9.0 - (df["health"] * 0.3) + np.random.RandomState(42).normal(0, 0.3, len(df))
df["sleep_hours"] = df["sleep_hours"].clip(5, 9)

# Assignment completion: UCI has paid (extra paid classes) + schoolsup as proxy
df["assignment_completion"] = np.clip(
    50 + df["schoolsup"].map({"yes": 20, "no": 0}) +
    df["famsup"].map({"yes": 15, "no": 0}) +
    df["paid"].map({"yes": 15, "no": 0}) +
    np.random.RandomState(42).normal(0, 8, len(df)), 20, 100
)

# Extracurricular: UCI has 'activities' (extra-curricular activities)
df["extracurricular"] = df["activities"].map({"yes": 1, "no": 0})

# Final score: G3 (0-20) -> scale to 0-100
df["final_score"] = df["G3"] * 5

# Select final columns
result = df[[
    "study_hours", "attendance", "previous_score", "sleep_hours",
    "assignment_completion", "extracurricular", "final_score"
]].round(2)

result.to_csv("data/students_real.csv", index=False)
print(f"\nSaved real dataset: {result.shape}")
print(result.head())
print(f"\nFinal score stats:")
print(result["final_score"].describe())
print(f"\nFeature correlations with final_score:")
print(result.corr()["final_score"].sort_values(ascending=False))