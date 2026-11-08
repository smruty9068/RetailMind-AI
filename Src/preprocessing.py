import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_data(df):
    """Clean the Telco Customer Churn dataset."""

    df = df.copy()

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df):
    """Create domain-specific features for customer churn prediction."""

    df = df.copy()

    # 1. Tenure-based features
    if "tenure" in df.columns:

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["New", "Short-term", "Medium-term", "Long-term"]
        )

    # 2. Charge-based features
    if {"MonthlyCharges", "TotalCharges", "tenure"}.issubset(df.columns):

        df["avg_monthly_charge"] = (
            df["TotalCharges"] /
            df["tenure"].replace(0, 1)
        )

        df["charge_ratio"] = (
            df["MonthlyCharges"] /
            (df["TotalCharges"] + 1)
        )

    # 3. Contract-based features
    if "Contract" in df.columns:

        df["is_month_to_month"] = (
            df["Contract"] == "Month-to-month"
        ).astype(int)

        df["is_one_year_contract"] = (
            df["Contract"] == "One year"
        ).astype(int)

        df["is_two_year_contract"] = (
            df["Contract"] == "Two year"
        ).astype(int)

    # 4. Service-based features
    service_columns = [
        "OnlineSecurity",
        "TechSupport",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies"
    ]

    for col in service_columns:
        if col in df.columns:
            df[f"has_{col.lower()}"] = (
                df[col] == "Yes"
            ).astype(int)

    if "InternetService" in df.columns:

        df["has_internet"] = (
            df["InternetService"] != "No"
        ).astype(int)

    available_services = [
        col for col in service_columns if col in df.columns
    ]

    if available_services:

        df["service_count"] = sum(
            (df[col] == "Yes").astype(int)
            for col in available_services
        )

    return df


def create_preprocessor(X):
    """Create preprocessing for numerical and categorical features."""

    num_cols = X.select_dtypes(include="number").columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor
