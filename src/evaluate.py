import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    test_data = np.load(DATA_PROCESSED_DIR / "test_split.npz", allow_pickle=True)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    with open(METRICS_DIR / "cv_training_summary.json", "r", encoding="utf-8") as f:
        cv_summary = json.load(f)

    model_names = ["logistic_regression", "naive_bayes", "svm", "mlp"]
    rows = []
    errors = {}

    plt.figure(figsize=(7, 5))
    has_roc = False

    pr_plt_data = []

    for name in model_names:
        model_path = METRICS_DIR / "trained_models" / f"{name}_best.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_val = roc_auc_score(y_test, y_score)

        p_curve, r_curve, _ = precision_recall_curve(y_test, y_score)
        pr_val = auc(r_curve, p_curve)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        cv_info = cv_summary.get(name, {})
        rows.append({
            "modelo": name,
            "cv_f1_mean": cv_info.get("cv_best_mean_score"),
            "cv_f1_std": cv_info.get("cv_best_std_score"),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_val), 4),
            "pr_auc": round(float(pr_val), 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

        errors[name] = {
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "total_errores": int(fp + fn),
            "tasa_error": round(float((fp + fn) / len(y_test)), 4),
        }

        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} ({roc_val:.3f})")
        pr_plt_data.append((name, r_curve, p_curve, pr_val))
        has_roc = True

    if has_roc:
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR (Recall)")
        plt.title("Curva ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, r_curve, p_curve, pr_val in pr_plt_data:
        plt.plot(r_curve, p_curve, label=f"{name} ({pr_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=300)
    plt.close()

    df_res = pd.DataFrame(rows)
    df_res.to_csv(METRICS_DIR / "evaluation_results.csv", index=False)

    with open(METRICS_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(df_res[["modelo", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print("\nResultados guardados en outputs/metrics/ y gráficas en outputs/figures/.")


if __name__ == "__main__":
    main()