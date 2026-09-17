# load -> preprocess -> train -> evaluate -> predict
import os
from src.data_preprocessing import preprocess
from src.evaluate import compare_models
from src.predict import predict_applicant, save_artifacts, select_best_model
from src.train_models import train

RAW_DATA_PATH = "data/raw/loan_data.csv"
PROCESSED_DATA_PATH = "data/processed/loan_data_cleaned.csv"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df, X_scaled, y, encoder, scaler = preprocess(RAW_DATA_PATH)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nFeatures shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")

    models, X_train, X_test, y_train, y_test = train(X_scaled, y)
    print(f"\nTrained {len(models)} models")
    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

    comparison = compare_models(models, X_test, y_test, results_dir=RESULTS_DIR)

    best_name, best_model = select_best_model(models, comparison, metric="F1-Score")
    save_artifacts(best_model, best_name, encoder, scaler, X_scaled.columns)

    sample_applicant = {
        "Age": 40,
        "Marital_Status": "Single",
        "Dependents": 0,
        "Education": "Graduate",
        "Employment_Type": "Salaried",
        "Income": 76500,
        "Credit_Score": 900,
        "Loan_Amount": 1020000,
        "Loan_Term": 36,
        "Existing_Debts": 5400,
        "Property_Area": "Semiurban",
    }
    prediction = predict_applicant(
        sample_applicant, best_model, encoder, scaler, X_scaled.columns
    )
    print(f"\nBest model by F1-Score: {best_name}")
    print("Sample applicant prediction:")
    print(prediction)

    


if __name__ == "__main__":
    main()
