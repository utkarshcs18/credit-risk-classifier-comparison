import os
import joblib
import pandas as pd

from src.data_preprocessing import CATEGORICAL_COLS, ID_COL, NUMERIC_COLS, TARGET_COL

ARTIFACTS_PATH = os.path.join("results", "credit_risk_pipeline.joblib")


def transform_applicant(applicant: dict, encoder, scaler, feature_columns) -> pd.DataFrame:
    df = pd.DataFrame([applicant])

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    encoded = encoder.transform(df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index,
    )
    df = df.drop(columns=CATEGORICAL_COLS).join(encoded_df)

    numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df[list(feature_columns)]


def predict_applicant(applicant: dict, model, encoder, scaler, feature_columns) -> dict:
    X = transform_applicant(applicant, encoder, scaler, feature_columns)
    predicted_class = int(model.predict(X)[0])
    default_probability = float(model.predict_proba(X)[0, 1])

    return {
        "predicted_class": predicted_class,
        "label": "risky/default" if predicted_class == 1 else "safe",
        "default_probability": round(default_probability, 4),
    }


def select_best_model(models: dict, comparison: pd.DataFrame, metric: str = "F1-Score"):
    best_name = comparison.loc[comparison[metric].idxmax(), "Model"]
    return best_name, models[best_name]


def save_artifacts(model, model_name, encoder, scaler, feature_columns, path: str = ARTIFACTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "encoder": encoder,
            "scaler": scaler,
            "feature_columns": list(feature_columns),
        },
        path,
    )
    print(f"Saved prediction pipeline ({model_name}) to {path}")


def load_artifacts(path: str = ARTIFACTS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved pipeline at {path}. Run python main.py first.")
    return joblib.load(path)


def predict_from_saved(applicant: dict, path: str = ARTIFACTS_PATH) -> dict:
    artifacts = load_artifacts(path)
    result = predict_applicant(
        applicant,
        artifacts["model"],
        artifacts["encoder"],
        artifacts["scaler"],
        artifacts["feature_columns"],
    )
    result["model_name"] = artifacts["model_name"]
    return result
