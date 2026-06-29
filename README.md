# Marketing ROI Optimization — Système Intelligent Multi-Modèles

---
title: Marketing ROI Optimizer
colorFrom: blue
colorTo: orange
sdk: streamlit
sdk_version: 1.45.1
app_file: dashboard/app.py
pinned: false
---

Projet M1 Data Engineering & AI — EFREI 2025-26  
Matière : Data Science / Machine Learning Supervisé

---

## Description

Plateforme complète d'optimisation du ROI marketing basée sur le dataset
[Dummy Advertising & Sales Data (Kaggle)](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).

Le système prédit les ventes générées par une combinaison de budgets publicitaires
(TV, Radio, Social Media, Influencer) et expose ces prédictions via un dashboard
interactif Streamlit et une API REST FastAPI.

---

## Structure du projet

```
marketing_roi_project/
├── data/
│   └── marketing_and_sales.csv     # Dataset Kaggle (à télécharger)
├── src/
│   ├── data_preparation.py         # EF1 — Pipeline preprocessing, EDA, feature engineering
│   ├── models.py                   # EF2 — 4 modèles ML/DL, GridSearch, timing
│   └── evaluation.py               # EF3 — Métriques, SHAP, courbes d'apprentissage, éco
├── api/
│   └── main.py                     # EF5 (optionnel) — API REST FastAPI
├── dashboard/
│   └── app.py                      # EF4 — Dashboard Streamlit (5 pages)
├── results/                        # Visualisations EDA, résidus, SHAP, courbes (généré)
├── saved_models/                   # Pipelines sérialisés .joblib (généré)
├── train.py                        # Script principal d'entraînement
├── requirements.txt                # Dashboard (Streamlit Cloud / Hugging Face)
└── requirements_full.txt           # Dashboard + API REST (usage local complet)
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd marketing_roi_project

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
# Dashboard uniquement (Streamlit Cloud) :
pip install -r requirements.txt
# Dashboard + API REST (usage local complet) :
pip install -r requirements_full.txt

# 4. Télécharger le dataset Kaggle
#    Depuis : https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data
#    Placer marketing_and_sales.csv dans le dossier data/
```

---

## Utilisation

### Étape 1 — Entraîner les modèles

```bash
python train.py
```

Ce script exécute dans l'ordre :
1. Chargement et nettoyage des données
2. EDA complète (distributions, corrélations, outliers, saturation)
3. Régression : Linear, Random Forest, Gradient Boosting, MLP
4. Classification (bonus) : Logistic, RF, GB, MLP
5. Évaluation : métriques, courbes d'apprentissage, SHAP, écoresponsabilité
6. GridSearchCV sur Gradient Boosting
7. Sauvegarde dans `saved_models/` et `results/`

Durée estimée : 2–5 minutes selon la machine.

### Étape 2 — Lancer le dashboard

```bash
python -m streamlit run dashboard/app.py
```

> **Important** : utiliser `python -m streamlit` (et non `streamlit` seul) pour garantir
> que le dashboard s'exécute avec le même interpréteur Python que celui utilisé pour
> entraîner les modèles. Un Python différent causerait une incompatibilité sklearn
> lors du chargement des fichiers `.joblib`.

Ouvre automatiquement `http://localhost:8501` dans le navigateur.

**Pages disponibles :**
| Page | Contenu |
|------|---------|
| Tableau de bord | KPIs globaux, budget par canal, top 10 campagnes ROI |
| Analyse des canaux | Corrélations, ROI marginal, recommandations par canal |
| Performance des modèles | Comparaison R²/MAE, importance des variables, SHAP global |
| Simulateur budgétaire | Prédiction temps réel, ROI, analyse de sensibilité |

Le dashboard utilise automatiquement l'API REST si elle est disponible (voyant vert
en haut à droite), sinon il charge le modèle local directement.

### Étape 3 (optionnel) — Lancer l'API REST

```bash
python -m uvicorn api.main:app --reload
```

Documentation interactive : `http://localhost:8000/docs` (Swagger)

---

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Santé du service + modèles disponibles |
| `POST` | `/predict` | Prédiction des ventes pour un scénario budgétaire |
| `GET` | `/model-info` | Informations sur les modèles entraînés |

**Exemple de requête `/predict` :**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TV": 80.0, "Radio": 18.0, "Social_Media": 5.0, "Influencer": "Macro", "model": "Gradient Boosting"}'
```

**Réponse :**
```json
{
  "predicted_sales_M": 284.71,
  "roi_estimate": 2.76,
  "total_budget_M": 103.0,
  "profit_estimate_M": 181.71,
  "model_used": "Gradient Boosting",
  "timestamp": "2026-06-22T..."
}
```

> Valeurs issues du modèle Gradient Boosting entraîné (R²=0.9985).
> TV domine les prédictions (corrélation r=0.999) : avec TV=80 M€,
> les ventes prédites atteignent 284.71 M€ — Radio et Social Media ont un impact marginal comparativement.

---

## Modèles implémentés

| Modèle | Type | Rôle | Score éco (entraînement) |
|--------|------|------|--------------------------|
| Linear / Logistic Regression | Baseline | Référence interprétable | ✅ Privilégier |
| Random Forest | Ensemble Bagging | Effets non-linéaires, FI native | 🟡 Acceptable |
| **Gradient Boosting** | Ensemble Boosting | **Modèle final recommandé** | ⚠️ Limiter |
| MLP (Deep Learning) | Réseau de neurones | Comparaison DL vs ML | ⚠️ Limiter |

> **Note** : le score éco mesure la consommation CPU à l'**entraînement** (proxy : temps de fit).
> Gradient Boosting est marqué "⚠️ Limiter" pour l'entraînement, mais reste le **modèle recommandé en production**
> grâce à son inference < 3 ms, son interprétabilité (SHAP, feature importance) et son meilleur R².
> Voir `results/eco_responsibility.csv` pour les scores détaillés.

---

## Résultats (régression — prédiction des ventes)

| Modèle | R² Test | MAE | RMSE | CV R² moy |
|--------|---------|-----|------|-----------|
| Linear Regression | 0.9960 | 2.59 | 5.88 | 0.9950 |
| Random Forest | 0.9984 | 2.63 | 3.72 | 0.9957 |
| **Gradient Boosting** | **0.9985** | **2.50** | **3.54** | **0.9949** |
| MLP Deep Learning | 0.9957 | 2.92 | 6.08 | 0.9946 |

> Les R² > 0.99 s'expliquent par le caractère synthétique du dataset (peu de bruit).
> Sur données réelles, R² attendu : 0.70–0.90.

---

## Fichiers générés après `python train.py`

```
results/
├── eda_distributions.png           # Distributions de toutes les variables
├── eda_correlation.png             # Heatmap corrélations
├── eda_scatter_budget_sales.png    # Impact budgets → ventes par influenceur
├── eda_boxplot_influencer.png      # Ventes par type d'influenceur
├── eda_saturation.png              # Rendement marginal décroissant (A15)
├── eda_multicollinearity.csv       # Analyse multicolinéarité (A23)
├── regression_comparison.csv       # Tableau comparatif régression
├── classification_comparison.csv   # Tableau comparatif classification
├── residuals_*.png                 # Résidus × 4 modèles régression
├── confusion_*.png                 # Matrices confusion × 4 modèles classification
├── regression_feature_importance.png
├── shap_summary.png                # SHAP global (Gradient Boosting)
├── shap_local_0.png                # SHAP local — prédiction MIN (C20)
├── shap_local_1.png                # SHAP local — prédiction MAX (C20)
├── learning_curve_reg_*.png        # Courbes d'apprentissage (B10)
├── worst_predictions_regression.csv # Pires prédictions (C15/C16)
├── train_test_gap_regression.csv   # Analyse overfitting (B11)
├── critical_r2_analysis.csv        # Analyse critique R² (C30)
└── eco_responsibility.csv          # Score écoresponsabilité (C29)

saved_models/
├── regression_*.joblib             # 4 pipelines régression
├── regression_Gradient_Boosting_tuned.joblib  # GB optimisé GridSearch (B15)
├── classification_*.joblib         # 4 pipelines classification
├── shap_data.joblib                # Valeurs SHAP pré-calculées
└── metadata.joblib                 # Métadonnées pour le dashboard
```

---

## Dépendances principales

```
pandas, numpy, scikit-learn, matplotlib, seaborn
shap, streamlit, plotly, joblib, scipy, requests
```

Voir [requirements.txt](requirements.txt) pour le dashboard, [requirements_full.txt](requirements_full.txt) pour l'installation complète avec API.
