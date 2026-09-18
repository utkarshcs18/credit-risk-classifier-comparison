import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def split_data(X, y, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def balance_train(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False,),
    }


def train_models(X_train, y_train) -> dict:
    trained = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"Fitted {name}")

    return trained


def train(X, y, test_size: float = 0.2):

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    print("\nClass counts before SMOTE:")
    print(y_train.value_counts())

    X_train_balanced, y_train_balanced = balance_train(X_train, y_train)

    print("\nClass counts after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    models = train_models(X_train_balanced, y_train_balanced)
    
    return models, X_train_balanced, X_test, y_train_balanced, y_test