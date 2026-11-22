from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE


def get_models_and_grids(random_state: int = RANDOM_STATE) -> dict:
    """Devuelve el modelo base y los tres modelos de clasificación"""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
        },
        "naive_bayes": {
            "estimator": GaussianNB(),
            "param_grid": {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        },
        "svm": {
            "estimator": SVC(
                probability=True,
                cache_size=512,
                random_state=random_state,
            ),
            "param_grid": [
                {
                    "kernel": ["linear"],
                    "C": [0.1, 1.0],
                },
                {
                    "kernel": ["rbf"],
                    "C": [0.1, 1.0, 10.0],
                    "gamma": [0.01, 0.1, 1.0],
                },
            ],
        },
        "mlp": {
            "estimator": MLPClassifier(
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=random_state,
            ),
            "param_grid": {
                "hidden_layer_sizes": [(64, 32), (128, 64)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 5e-4],
            },
        },
    }