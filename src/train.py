import json
import time
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import (
    CV_SPLITS,
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    RANDOM_STATE,
)
from src.data_prep import prepare_datasets
from src.models import get_models_and_grids


def load_training_data():
    """Carga los datos preparados desde disco o los genera si aún no existen."""
    npz_path = DATA_PROCESSED_DIR / "bank_marketing_prepared.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return (
            data["X_train"],
            data["X_test"],
            data["y_train"],
            data["y_test"],
            data["feature_names"],
        )

    X_train, X_test, y_train, y_test, feature_names, _ = prepare_datasets(
        save_processed=True
    )
    return X_train, X_test, y_train, y_test, feature_names


def train_all_models(
    scoring: str = "f1",
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    selected_models: list[str] | None = None,
    svm_max_samples: int = 10000,
):
    """Entrena y ajusta con validación cruzada el modelo base y los tres modelos."""
    X_train, X_test, y_train, y_test, feature_names = load_training_data()
    models_config = get_models_and_grids(random_state=random_state)

    if selected_models is not None:
        models_config = {
            name: cfg
            for name, cfg in models_config.items()
            if name in selected_models
        }

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    models_dir = METRICS_DIR / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    training_summary = {}

    for model_name, cfg in models_config.items():
        print(f"\n--- Entrenando y ajustando: {model_name} ---")
        start_time = time.time()

        X_train_fit, y_train_fit = X_train, y_train
        n_jobs_cv = -1
        used_samples = len(X_train)

        if model_name == "svm":
            n_jobs_cv = 1
            if len(X_train) > svm_max_samples:
                X_train_fit, _, y_train_fit, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=svm_max_samples,
                    stratify=y_train,
                    random_state=random_state,
                )
                used_samples = len(X_train_fit)
                print(
                    f"[svm] Usando submuestreo estratificado de {used_samples} muestras "
                    f"para evitar tiempos excesivos de ajuste."
                )

        grid_search = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs_cv,
            refit=True,
            return_train_score=True,
        )
        grid_search.fit(X_train_fit, y_train_fit)

        elapsed_sec = round(time.time() - start_time, 2)
        best_idx = grid_search.best_index_
        cv_mean = float(grid_search.cv_results_["mean_test_score"][best_idx])
        cv_std = float(grid_search.cv_results_["std_test_score"][best_idx])

        best_model_path = models_dir / f"{model_name}_best.joblib"
        joblib.dump(grid_search.best_estimator_, best_model_path)

        training_summary[model_name] = {
            "scoring": scoring,
            "cv_splits": cv_splits,
            "samples_used_in_search": used_samples,
            "best_params": grid_search.best_params_,
            "cv_best_mean_score": cv_mean,
            "cv_best_std_score": cv_std,
            "search_space": cfg["param_grid"],
            "training_time_sec": elapsed_sec,
            "model_path": str(best_model_path),
        }

        print(
            f"[{model_name}] Mejor {scoring} CV: {cv_mean:.4f} (+/- {cv_std:.4f}) "
            f"| Tiempo: {elapsed_sec}s"
        )
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")

    eval_pack_path = DATA_PROCESSED_DIR / "test_split.npz"
    np.savez_compressed(
        eval_pack_path,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )

    summary_path = METRICS_DIR / "cv_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResumen guardado en: {summary_path}")
    return training_summary


if __name__ == "__main__":
    train_all_models(selected_models=["svm"])