import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score


CM_FILENAMES = {
    "Logistic Regression": "logistic_regression_cm.png",
    "Decision Tree": "decision_tree_cm.png",
    "SVM": "svm_cm.png",
}


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }


def save_confusion_matrix(model, X_test, y_test, save_path: str, title: str) -> None:
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Safe (0)", "Default (1)"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path: str) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def compare_models(models: dict, X_test, y_test, results_dir: str = "results") -> pd.DataFrame:
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})

        filename = CM_FILENAMES.get(name, f"{name.lower().replace(' ', '_')}_cm.png")
        save_confusion_matrix(
            model,
            X_test,
            y_test,
            save_path=os.path.join(cm_dir, filename),
            title=f"{name} Confusion Matrix",
        )

    comparison = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    comparison[metric_cols] = comparison[metric_cols].round(4)

    csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    comparison.to_csv(csv_path, index=False)

    print("\nModel comparison (metrics for Default = 1):")
    print(comparison.to_string(index=False))
    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved confusion matrices to {cm_dir}/")

    tree = models.get("Decision Tree")
    if tree is not None and hasattr(tree, "feature_importances_"):
        feature_names = list(X_test.columns)
        importance_path = os.path.join(results_dir, "feature_importance.png")
        plot_feature_importance(tree, feature_names, importance_path)
        print(f"Saved feature importance to {importance_path}")

    return comparison
