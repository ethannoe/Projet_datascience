"""
Définition et entraînement des modèles (régression et classification).

Justification du choix des 4 modèles (régression et classification) :

1. Linear / Logistic Regression — BASELINE
   Hypothèse de linéarité, très interprétable (coefficients directement lisibles).
   Sert de référence : si les modèles complexes ne font pas mieux, inutile de les déployer.

2. Random Forest — ENSEMBLE BAGGING
   - Capture les effets non-linéaires et les interactions entre canaux (TV × Social Media)
   - Robuste aux outliers (vote par majorité)
   - Fournit une feature importance native (réduction d'impureté / Gini)
   - Peu sensible au scaling → même pipeline, les transformations scalent les autres modèles

3. Gradient Boosting — ENSEMBLE BOOSTING
   - Apprend séquentiellement les erreurs résiduelles → meilleure précision que RF sur données
     avec patterns subtils (rendement marginal décroissant)
   - Hyperparamètres interprétables (learning_rate / depth contrôlent biais-variance)
   - Souvent le meilleur modèle sur des datasets tabulaires de taille modeste (~200 lignes)

4. MLPRegressor / MLPClassifier — DEEP LEARNING
   - Réseau de neurones multicouche (3 couches cachées : 128-64-32)
   - Capable de modéliser des interactions complexes non linéaires entre canaux et influenceurs
   - early_stopping=True évite l'overfitting sur 200 enregistrements
   - Sur un dataset aussi petit, le MLP risque de ne pas dominer GB, ce qui sera analysé
     (démonstration que DL n'est pas toujours supérieur — objectif pédagogique explicite)

Remarque architecturale MLP :
   input(6) → 128 → 64 → 32 → output(1)
   - 3 couches cachées = réseau "profond" pour notre contexte
   - ReLU activation : évite le vanishing gradient, bonne convergence
   - Adam optimizer : adaptatif, moins sensible au learning rate initial
   - validation_fraction=0.15 pour early stopping interne
"""

import os
import time
import joblib
from typing import Dict

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

REG_MODEL_NAMES = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "MLP (Deep Learning)",
]
CLF_MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "MLP (Deep Learning)",
]


def _filename(task: str, name: str) -> str:
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{task}_{safe}.joblib"


def build_regression_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_split=4,
                random_state=RANDOM_STATE,
            )),
        ]),
        "MLP (Deep Learning)": Pipeline([
            ("preprocessor", preprocessor),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def build_classification_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "MLP (Deep Learning)": Pipeline([
            ("preprocessor", preprocessor),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=2000,
                # early_stopping incompatible avec numpy 2.x + labels texte (np.isnan sur str)
                early_stopping=False,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def train_all_models(
    models: Dict[str, Pipeline],
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "r2",
) -> Dict:
    """
    Entraîne chaque pipeline et calcule la cross-validation.
    Retourne un dict {nom: {pipeline, cv_scores, cv_mean, cv_std, train_time_s}}.
    train_time_s est utilisé pour l'analyse écoresponsabilité (B12/C29).
    """
    results = {}
    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 3)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        results[name] = {
            "pipeline":    pipeline,
            "cv_scores":   cv_scores,
            "cv_mean":     float(cv_scores.mean()),
            "cv_std":      float(cv_scores.std()),
            "train_time_s": elapsed,
        }
        print(f"  {name:<30} CV {scoring} = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  [{elapsed:.2f}s]")
    return results


def save_models(results: Dict, task: str, output_dir: str = "saved_models") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, info in results.items():
        path = os.path.join(output_dir, _filename(task, name))
        joblib.dump(info["pipeline"], path)


def load_all_models(task: str, names: list, model_dir: str = "saved_models") -> Dict:
    models = {}
    for name in names:
        path = os.path.join(model_dir, _filename(task, name))
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


# ─── B15/B16 : GridSearchCV sur Gradient Boosting ────────────────────────────

def run_grid_search(X_train, y_train, preprocessor: ColumnTransformer,
                    task: str = "regression", cv: int = 5):
    """
    GridSearchCV sur Gradient Boosting — modèle candidat optimal pour ce dataset.

    Stratégie d'optimisation (B16) :
    - n_estimators et learning_rate contrôlent directement le biais/variance trade-off.
    - max_depth contrôle la complexité individuelle des arbres (risque d'overfitting).
    - Grille 3×3×3 = 27 combinaisons × 5-fold CV = 135 fits — raisonnable sur ~200 obs.
    - subsample=0.8 fixé (régularisation stochastique, hors grille pour limiter le temps).
    - refit=True : le meilleur pipeline est réentraîné sur tout X_train et prêt à l'emploi.
    """
    from sklearn.model_selection import GridSearchCV

    if task == "regression":
        base = GradientBoostingRegressor(subsample=0.8, min_samples_split=4, random_state=RANDOM_STATE)
        scoring = "r2"
    else:
        base = GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE)
        scoring = "f1_weighted"

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base)])

    param_grid = {
        "model__n_estimators":  [100, 200, 300],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth":     [3, 4, 5],
    }

    gs = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1, verbose=0, refit=True,
        return_train_score=True,
    )
    t0 = time.time()
    gs.fit(X_train, y_train)
    print(f"  GridSearch terminé en {time.time() - t0:.1f}s")
    return gs


# ─── B21/B22 : Complexité de déploiement par modèle ─────────────────────────

DEPLOYMENT_COMPLEXITY: Dict[str, Dict] = {
    "Linear Regression":   {"params_est": "~6",    "inference_ms": "< 1",  "export": "joblib / ONNX", "note": "✅ Idéal prod"},
    "Logistic Regression": {"params_est": "~18",   "inference_ms": "< 1",  "export": "joblib / ONNX", "note": "✅ Idéal prod"},
    "Random Forest":       {"params_est": "~200k", "inference_ms": "1–5",  "export": "joblib",         "note": "🟡 Acceptable"},
    "Gradient Boosting":   {"params_est": "~20k",  "inference_ms": "1–3",  "export": "joblib / ONNX", "note": "✅ Recommandé"},
    "MLP (Deep Learning)": {"params_est": "~24k",  "inference_ms": "1–5",  "export": "joblib / ONNX", "note": "🟡 Acceptable"},
}
