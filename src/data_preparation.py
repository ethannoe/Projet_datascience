"""
Acquisition et préparation des données marketing.

Responsabilités :
  - Chargement et nettoyage du dataset Kaggle (marketing_and_sales.csv)
  - Feature engineering métier (Total_Budget, ROI, parts de budget, classes de performance)
  - Construction du pipeline de préprocessing sklearn sans data leakage
  - Splits train/test (régression et classification)

Choix techniques :
  - StandardScaler sur les numériques : centre et réduit pour ne pas pénaliser MLP et LR
  - OneHotEncoder avec drop='first' : évite la multicolinéarité parfaite (référence = 'Mega')
  - ColumnTransformer : garantit que le scaling est ajusté UNIQUEMENT sur le train set
  - Stratified split pour la classification : préserve la distribution des classes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

NUMERIC_FEATURES = ["TV", "Radio", "Social_Media"]
CATEGORICAL_FEATURES = ["Influencer"]
INFLUENCER_CATEGORIES = [["Mega", "Macro", "Micro", "Nano"]]
TARGET_REG = "Sales"
TARGET_CLF = "Performance"
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # La colonne Kaggle s'appelle "Social Media" — on normalise le nom
    df = df.rename(columns={"Social Media": "Social_Media"})
    # Suppression des lignes sans cible (Sales)
    df = df.dropna(subset=["Sales"])
    # Imputation médiane/mode sur les features
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df["Influencer"] = df["Influencer"].fillna(df["Influencer"].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dérivées utilisées pour l'EDA et le dashboard.
    Ces features NE sont PAS toutes injectées dans les modèles (risque de
    multicolinéarité pour LR) ; elles servent à l'analyse exploratoire et
    au calcul du ROI dans le simulateur budgétaire.
    """
    df = df.copy()
    df["Total_Budget"] = df["TV"] + df["Radio"] + df["Social_Media"]
    safe_total = df["Total_Budget"].replace(0, np.nan)
    df["ROI"] = df["Sales"] / safe_total
    df["TV_Share"] = df["TV"] / safe_total
    df["Radio_Share"] = df["Radio"] / safe_total
    df["Social_Share"] = df["Social_Media"] / safe_total
    # Classes de performance pour la tâche de classification (bonus)
    # Découpage en terciles pour garantir 3 classes équilibrées
    q33 = df["Sales"].quantile(1 / 3)
    q66 = df["Sales"].quantile(2 / 3)
    df[TARGET_CLF] = pd.cut(
        df["Sales"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Pipeline de préprocessing réutilisable (instancié à chaque tâche pour
    éviter le partage d'état entre régression et classification).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=INFLUENCER_CATEGORIES,
                    drop="first",           # référence = 'Mega', évite la colinéarité
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def split_regression(df: pd.DataFrame, test_size: float = 0.2):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_REG]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def split_classification(df: pd.DataFrame, test_size: float = 0.2):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_CLF].astype(str)
    # Stratification obligatoire pour ne pas déséquilibrer les classes au split
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def get_feature_names_after_ohe(preprocessor) -> list:
    """Retourne les noms de colonnes après transformation (utile pour SHAP et FI)."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


# ─── A10 : Détection des outliers (IQR) ──────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Détecte les outliers par la méthode IQR (Tukey).
    Borne inf = Q1 - 1.5×IQR  |  Borne sup = Q3 + 1.5×IQR
    Retourne un tableau résumant bornes, nombre et % d'outliers par variable.
    """
    if columns is None:
        columns = NUMERIC_FEATURES + ["Sales"]
    records = []
    for col in [c for c in columns if c in df.columns]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        records.append({
            "Variable":    col,
            "Q1":          round(Q1, 3),
            "Q3":          round(Q3, 3),
            "IQR":         round(IQR, 3),
            "Borne inf.":  round(lower, 3),
            "Borne sup.":  round(upper, 3),
            "N outliers":  n_out,
            "% outliers":  round(n_out / len(df) * 100, 1),
        })
    return pd.DataFrame(records)


# ─── A12 : Variables quasi-constantes ────────────────────────────────────────

def check_quasi_constant(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Identifie les variables quasi-constantes (une valeur domine > threshold).
    Une variable quasi-constante n'apporte aucune information discriminante au modèle.
    """
    records = []
    for col in df.select_dtypes(include=[np.number]).columns:
        vc = df[col].value_counts(normalize=True)
        records.append({
            "Variable":        col,
            "Valeur dominante": vc.index[0],
            "Fréquence":       round(float(vc.iloc[0]), 4),
            "Quasi-constante": bool(vc.iloc[0] > threshold),
        })
    return pd.DataFrame(records).sort_values("Fréquence", ascending=False)


# ─── A23 : Multicolinéarité ───────────────────────────────────────────────────

def check_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la multicolinéarité entre features via la matrice de corrélation.
    Signale les paires avec |r| > 0.7 (seuil modéré pour être exhaustif).

    Note architecturale : Total_Budget = TV + Radio + Social_Media par construction
    → corrélation mécanique élevée attendue et non pathologique.
    C'est pourquoi Total_Budget est EXCLU des features du modèle (NUMERIC_FEATURES
    ne contient que TV, Radio, Social_Media), évitant la multicolinéarité parfaite.
    """
    num_cols = [c for c in NUMERIC_FEATURES + ["Total_Budget", "Sales"] if c in df.columns]
    corr = df[num_cols].corr()
    records = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) > 0.7:
                records.append({
                    "Variable 1":            cols[i],
                    "Variable 2":            cols[j],
                    "Corrélation r":         round(r, 4),
                    "Niveau":                "ÉLEVÉ (> 0.9)" if abs(r) > 0.9 else "MODÉRÉ (0.7–0.9)",
                    "Risque multicolinéarité": abs(r) > 0.9,
                })
    return pd.DataFrame(records).sort_values("Corrélation r", key=abs, ascending=False)


# ─── A15 : Points de saturation budgétaire ───────────────────────────────────

def find_saturation_points(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Détecte les points de saturation (rendement marginal décroissant) par canal.
    Méthode : ROI empirique par décile de budget.
    Saturation = premier décile où le ROI chute de plus de 15% par rapport au ROI peak.
    """
    records = []
    for channel in NUMERIC_FEATURES:
        if channel not in df.columns:
            continue
        roi_col = df["Sales"] / df[channel].replace(0, np.nan)
        df_temp = pd.DataFrame({channel: df[channel], "roi": roi_col}).dropna()
        try:
            df_temp["bin"] = pd.qcut(df_temp[channel], q=n_bins, duplicates="drop")
        except ValueError:
            records.append({"Canal": channel, "Budget saturation estimé (M€)": "Non calculable", "ROI max observé": None})
            continue
        grouped = df_temp.groupby("bin", observed=True).agg(
            budget_mean=(channel, "mean"),
            roi_mean=("roi", "mean"),
        ).reset_index()
        peak_roi = float(grouped["roi_mean"].max())
        mask = grouped["roi_mean"] < peak_roi * 0.85
        sat_budget = float(grouped.loc[mask.idxmax(), "budget_mean"]) if mask.any() else None
        records.append({
            "Canal":                        channel,
            "Budget saturation estimé (M€)": round(sat_budget, 1) if sat_budget else "Non détecté",
            "ROI max observé":              round(peak_roi, 2),
        })
    return pd.DataFrame(records)
