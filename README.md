# Linear Regression Summative: Student Performance Prediction 

## Mission & Problem

My mission is to improve access to quality education for young people in Africa by helping
schools identify students at risk of underperforming before final exams. This project predicts
a student's final Mathematics grade (G3, 0–20) from academic history and study habits, so
schools can target tutoring and support early. Dataset: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (395 students, regression target G3).

## Public API endpoint

- **Swagger UI:** https://studentgradeapi.onrender.com/docs  
- **Prediction endpoint:** `POST https://studentgradeapi.onrender.com/predict`
- **Retraining endpoint:** `POST https://studentgradeapi.onrender.com/retrain` (upload a CSV of new student rows)

Example request body for `/predict`:

```json
{
  "G1": 12, "G2": 13, "failures": 0, "Medu": 3,
  "age": 16, "goout": 3, "studytime": 2, "absences": 4
}
```

## Video demo: (https://youtu.be/qEzk_RNLm60)

## Repository structure

```
summative/
├── linear_regression/
│   ├── multivariate.ipynb   # EDA, feature engineering, SGD vs Decision Tree vs Random Forest
│   ├── predict.py           # script that predicts with the saved best model
│   └── data/student-mat.csv
├── API/
│   ├── prediction.py        # FastAPI app (POST /predict, POST /retrain, CORS)
│   └── models/best_model.pkl
└── FlutterApp/              # one-page mobile app calling the public API
pyproject.toml / uv.lock / requirements.txt
```

## How to run

### 1. Notebook & prediction script (uses [uv](https://docs.astral.sh/uv/))

```bash
uv sync                                             # creates .venv and installs everything
uv run jupyter notebook summative/linear_regression/multivariate.ipynb
uv run python summative/linear_regression/predict.py --G1 12 --G2 13 --failures 0 \
    --Medu 3 --age 16 --goout 3 --studytime 2 --absences 4
```

### 2. API locally

```bash
uv run uvicorn prediction:app --reload --app-dir summative/API
# Swagger UI at http://127.0.0.1:8000/docs
```

Deployment on Render: build command `pip install -r requirements.txt`, start command
`uvicorn summative.API.prediction:app --host 0.0.0.0 --port $PORT`, and set the
environment variable `PYTHON_VERSION=3.13.4` so the pickled model loads under the same
Python version it was trained with.

### 3. Mobile app

Requirements: [Flutter SDK](https://docs.flutter.dev/get-started/install) and an Android
emulator or physical device (USB debugging enabled).

```bash
cd summative/FlutterApp
flutter pub get
flutter run          # pick your connected Android device when prompted
```

The API base URL is set in `lib/main.dart` (`apiBaseUrl` constant) and points at the public
Render service. Enter the eight input values, press **Predict**, and the predicted final
grade (or a validation error for missing/out-of-range values) is displayed on the page.
