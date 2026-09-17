import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "Applicant_ID"
TARGET_COL = "Default"
CATEGORICAL_COLS = ["Marital_Status", 
                    "Education", 
                    "Employment_Type", 
                    "Property_Area"
                    ]

NUMERIC_COLS = [
    "Age",
    "Dependents",
    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
    "Existing_Debts",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    print("Sample Data:")
    print(df.head())

    print("\nDimensions:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nInfo:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())



def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Marital_Status", "Employment_Type"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].median()).astype(int)

    for col in ["Income", "Credit_Score", "Existing_Debts"]:
        df[col] = df[col].fillna(df[col].mean())

    return df


def encode_categorical(df: pd.DataFrame, columns: list[str] | None = None):
    df = df.copy()
    columns = columns or CATEGORICAL_COLS

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index,
    )

    return df.drop(columns=columns).join(encoded_df), encoder


def feature_scaling(df: pd.DataFrame, features: list[str] | None = None):
    df_scaled = df.copy()
    features = features or [col for col in NUMERIC_COLS if col in df_scaled.columns]

    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])
    return df_scaled, scaler


def preprocess(raw_path: str):
    df = load_data(raw_path)
    inspect_data(df)

    df = drop_id(df)
    df = handle_missing(df)
    df, encoder = encode_categorical(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    X_scaled, scaler = feature_scaling(X)

    return df, X_scaled, y, encoder, scaler
