"""Análisis de errores: qué clientes se equivocan los modelos y por qué.

Requiere haber corrido antes `python -m src.train`.
Uso: python -m src.error_analysis
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR
from src.data_prep import load_and_split_data

MODEL_NAMES = ["logistic_regression", "naive_bayes", "svm", "mlp"]
CAT_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
NUM_COLS = ["age", "balance", "duration", "campaign", "pdays", "previous"]
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
MIN_POSITIVES = 30  # grupos con menos positivos reales son muy ruidosos


def load_test_with_predictions():
    """Reconstruye el X_test crudo (con sus categorías) y agrega las predicciones."""
    # El split es determinista (misma semilla), así que coincide con test_split.npz
    _, X_test_raw, _, y_test_raw = load_and_split_data()
    saved = np.load(DATA_PROCESSED_DIR / "test_split.npz", allow_pickle=True)
    X_test, y_test = saved["X_test"], saved["y_test"]

    if not np.array_equal(y_test_raw.to_numpy(), y_test):
        raise RuntimeError(
            "El test crudo no coincide con test_split.npz. "
            "Vuelve a correr data_prep y train."
        )

    df = X_test_raw.reset_index(drop=True).copy()
    df["y_true"] = y_test
    for name in MODEL_NAMES:
        path = METRICS_DIR / "trained_models" / f"{name}_best.joblib"
        df[f"pred_{name}"] = joblib.load(path).predict(X_test)
    return df


def outcome(y_true, y_pred):
    """Etiqueta cada fila como TP, FN, FP o TN."""
    return np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 1) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1)],
        ["TP", "FN", "FP"],
        default="TN",
    )


def errors_by_group(df):
    """Recall y tasa de falsos positivos por categoría de cada variable."""
    rows = []
    for name in MODEL_NAMES:
        kind = outcome(df["y_true"].to_numpy(), df[f"pred_{name}"].to_numpy())
        for col in CAT_COLS:
            for cat, idx in df.groupby(col).groups.items():
                k = pd.Series(kind, index=df.index).loc[idx]
                tp, fn = (k == "TP").sum(), (k == "FN").sum()
                fp, tn = (k == "FP").sum(), (k == "TN").sum()
                rows.append({
                    "modelo": name, "variable": col, "categoria": cat,
                    "n": len(idx), "positivos": tp + fn, "negativos": fp + tn,
                    "fn": fn, "fp": fp,
                    "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                    "tasa_fp": fp / (fp + tn) if (fp + tn) else np.nan,
                })
    return pd.DataFrame(rows)


def numeric_profile(df):
    """Compara variables numéricas entre TP, FN, FP y TN."""
    rows = []
    for name in MODEL_NAMES:
        df_k = df.assign(grupo=outcome(df["y_true"].to_numpy(),
                                       df[f"pred_{name}"].to_numpy()))
        for col in NUM_COLS:
            for grupo, sub in df_k.groupby("grupo")[col]:
                rows.append({
                    "modelo": name, "variable": col, "grupo": grupo,
                    "n": int(sub.count()),
                    "media": round(float(sub.mean()), 2),
                    "mediana": round(float(sub.median()), 2),
                })
    return pd.DataFrame(rows)


def shared_errors(df):
    """Casos que todos los modelos fallan (más difíciles) y cuántos fallan solo uno."""
    y = df["y_true"].to_numpy()
    wrong = np.column_stack([df[f"pred_{m}"].to_numpy() != y for m in MODEL_NAMES])
    pos = y == 1
    return {
        "positivos_que_ningun_modelo_detecta": int((wrong.all(axis=1) & pos).sum()),
        "positivos_detectados_por_todos": int((~wrong.any(axis=1) & pos).sum()),
        "total_positivos": int(pos.sum()),
        "falsos_positivos_de_todos_los_modelos": int((wrong.all(axis=1) & ~pos).sum()),
        "total_negativos": int((~pos).sum()),
    }


def plot_recall_by_group(groups):
    """Recall por categoría para poutcome, month y job (todos los modelos)."""
    variables = ["poutcome", "month", "job"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, var in zip(axes, variables):
        sub = groups[(groups["variable"] == var) & (groups["positivos"] >= MIN_POSITIVES)]
        cats = sub["categoria"].unique().tolist()
        if var == "month":
            cats = [m for m in MONTH_ORDER if m in cats]
        else:
            cats = sorted(cats)
        width = 0.8 / len(MODEL_NAMES)
        x = np.arange(len(cats))
        for i, name in enumerate(MODEL_NAMES):
            vals = [
                sub[(sub["modelo"] == name) & (sub["categoria"] == c)]["recall"].iloc[0]
                for c in cats
            ]
            ax.bar(x + i * width, vals, width, label=name)
        ax.set_xticks(x + width * (len(MODEL_NAMES) - 1) / 2)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title(f"Recall por {var}")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_recall_by_group.png", dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_test_with_predictions()
    groups = errors_by_group(df)
    numeric = numeric_profile(df)
    shared = shared_errors(df)

    groups.to_csv(METRICS_DIR / "error_by_group.csv", index=False)
    numeric.to_csv(METRICS_DIR / "error_numeric_profile.csv", index=False)
    with open(METRICS_DIR / "error_shared.json", "w", encoding="utf-8") as f:
        json.dump(shared, f, indent=2, ensure_ascii=False)
    plot_recall_by_group(groups)

    print("== Casos difíciles (comunes a los 4 modelos) ==")
    for k, v in shared.items():
        print(f"{k}: {v}")

    print(f"\n== Grupos con peor recall (>= {MIN_POSITIVES} positivos reales) ==")
    for name in MODEL_NAMES:
        sub = groups[(groups["modelo"] == name) & (groups["positivos"] >= MIN_POSITIVES)]
        worst = sub.sort_values("recall").head(3)
        txt = ", ".join(
            f"{r.variable}={r.categoria} (recall {r.recall:.2f}, n+={r.positivos})"
            for r in worst.itertuples()
        )
        print(f"{name}: {txt}")

    print("\n== duration: mediana por tipo de resultado ==")
    dur = numeric[numeric["variable"] == "duration"].pivot(
        index="modelo", columns="grupo", values="mediana"
    )
    print(dur.to_string())

    print("\nGuardado en outputs/metrics/ (error_by_group.csv, error_numeric_profile.csv, "
          "error_shared.json) y outputs/figures/error_recall_by_group.png")


if __name__ == "__main__":
    main()