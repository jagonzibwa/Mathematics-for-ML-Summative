"""Make a final-grade prediction with the best-performing saved model.

Usage:
    uv run python summative/linear_regression/predict.py
    uv run python summative/linear_regression/predict.py --G1 8 --G2 7 --failures 2 \
        --Medu 1 --age 18 --goout 4 --studytime 1 --absences 12
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "API" / "models" / "best_model.pkl"


def predict_final_grade(G1: float, G2: float, failures: int, Medu: int, age: int,
                        goout: int, studytime: int, absences: int) -> float:
    """Predict a student's final grade G3 (0-20) using the saved best model."""
    bundle = joblib.load(BUNDLE_PATH)
    row = pd.DataFrame(
        [[G1, G2, failures, Medu, age, goout, studytime, absences]],
        columns=bundle["features"],
    )
    prediction = bundle["model"].predict(bundle["scaler"].transform(row))[0]
    return float(np.clip(prediction, 0, 20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a student's final grade (G3).")
    parser.add_argument("--G1", type=float, default=12, help="First-period grade (0-20)")
    parser.add_argument("--G2", type=float, default=13, help="Second-period grade (0-20)")
    parser.add_argument("--failures", type=int, default=0, help="Past class failures (0-4)")
    parser.add_argument("--Medu", type=int, default=3, help="Mother's education level (0-4)")
    parser.add_argument("--age", type=int, default=16, help="Student age (15-22)")
    parser.add_argument("--goout", type=int, default=3, help="Going out with friends (1-5)")
    parser.add_argument("--studytime", type=int, default=2, help="Weekly study time (1-4)")
    parser.add_argument("--absences", type=int, default=4, help="School absences (0-93)")
    args = parser.parse_args()

    grade = predict_final_grade(**vars(args))
    print(f"Predicted final grade (G3): {grade:.2f} / 20")
