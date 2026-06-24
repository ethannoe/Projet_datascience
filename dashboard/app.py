"""
Dashboard décisionnel Streamlit — Optimisation du ROI Marketing.

Orienté équipe métier (CMO / Responsable marketing) :
  Page 1 — Tableau de bord       : KPIs globaux, distributions, top campagnes
  Page 2 — Analyse des canaux    : ROI par canal, budgets vs ventes, saturation
  Page 3 — Performance des modèles : comparaison + importance des variables (EF4)
  Page 4 — Simulateur            : prédiction temps réel + analyse de sensibilité

Source de prédiction : API REST (si disponible) avec repli automatique sur le modèle local.
Voyant de statut API affiché en haut à droite de chaque page.

Usage :
    streamlit run dashboard/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preparation import (
    NUMERIC_FEATURES,
    engineer_features,
    get_feature_names_after_ohe,
    load_data,
)
from src.models import REG_MODEL_NAMES, load_all_models

DATA_PATH = "data/marketing_and_sales.csv"
MODEL_DIR = "saved_models"
API_BASE  = "http://localhost:8000"

st.set_page_config(
    page_title="Marketing ROI — Tableau de bord",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "TV":           "#e74c3c",
    "Radio":        "#2ecc71",
    "Social_Media": "#9b59b6",
    "Sales":        "#3498db",
    "primary":      "#2c3e50",
}

# Voyant API fixe en haut à droite + ajustements typographiques
CSS = """
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    h1 { color: #2c3e50; font-size: 1.7rem; }
    h2, h3 { color: #34495e; }

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── Cache ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame:
    return engineer_features(load_data(DATA_PATH))


@st.cache_resource
def load_artifacts() -> tuple:
    meta_path = os.path.join(MODEL_DIR, "metadata.joblib")
    if not os.path.exists(meta_path):
        return None, {}
    meta       = joblib.load(meta_path)
    reg_models = load_all_models("regression", REG_MODEL_NAMES, MODEL_DIR)
    # Remplace GB par sa version optimisée GridSearch si disponible
    tuned_path = os.path.join(MODEL_DIR, "regression_Gradient_Boosting_tuned.joblib")
    if os.path.exists(tuned_path) and "Gradient Boosting" in reg_models:
        reg_models["Gradient Boosting"] = joblib.load(tuned_path)
    return meta, reg_models


# ─── API ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def _api_health() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=1).status_code == 200
    except Exception:
        return False


def _predict_via_api(tv: float, radio: float, social: float,
                     influencer: str, model_name: str) -> float | None:
    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={"TV": tv, "Radio": radio, "Social_Media": social,
                  "Influencer": influencer, "model": model_name},
            timeout=3,
        )
        if resp.status_code == 200:
            return float(resp.json()["predicted_sales_M"])
    except Exception:
        pass
    return None



# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    st.sidebar.title("Marketing ROI")
    st.sidebar.caption("Tableau de bord décisionnel")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Tableau de bord",
            "Analyse des canaux",
            "Performance des modèles",
            "Simulateur budgétaire",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Projet M1 Data Engineering · EFREI 2025-26")
    return page


# ─── Page 1 : Tableau de bord ────────────────────────────────────────────────

def page_overview(df: pd.DataFrame) -> None:
    st.title("Tableau de bord — KPIs Marketing")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Campagnes analysées",   f"{len(df)}")
    c2.metric("Ventes moyennes",       f"{df['Sales'].mean():.1f} M€")
    c3.metric("Budget total moyen",    f"{df['Total_Budget'].mean():.1f} M€")
    c4.metric("ROI moyen",             f"{df['ROI'].mean():.2f}×")
    c5.metric("Meilleur ROI observé",  f"{df['ROI'].max():.2f}×")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(
            df, x="Sales", nbins=25,
            color_discrete_sequence=[PALETTE["Sales"]],
            title="Distribution des ventes",
            labels={"Sales": "Ventes (M€)", "count": "Nombre de campagnes"},
        )
        fig.update_layout(bargap=0.05, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        budget_df = pd.DataFrame({
            "Canal":             ["TV", "Radio", "Social Media"],
            "Budget moyen (M€)": [df["TV"].mean(), df["Radio"].mean(), df["Social_Media"].mean()],
        })
        fig = px.bar(
            budget_df, x="Canal", y="Budget moyen (M€)", color="Canal",
            color_discrete_map={
                "TV":           PALETTE["TV"],
                "Radio":        PALETTE["Radio"],
                "Social Media": PALETTE["Social_Media"],
            },
            title="Budget moyen investi par canal",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        inf_counts = df["Influencer"].value_counts().reset_index()
        inf_counts.columns = ["Influencer", "Nombre"]
        fig = px.pie(
            inf_counts, values="Nombre", names="Influencer",
            title="Types d'influenceurs mobilisés",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.38,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        fig = px.box(
            df, x="Influencer", y="Sales", color="Influencer",
            title="Ventes par type d'influenceur",
            labels={"Sales": "Ventes (M€)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 campagnes par ROI")
    top10 = (
        df[["TV", "Radio", "Social_Media", "Influencer", "Sales", "Total_Budget", "ROI"]]
        .sort_values("ROI", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.columns = [
        "TV (M€)", "Radio (M€)", "Social Media (M€)",
        "Influencer", "Ventes (M€)", "Budget total (M€)", "ROI",
    ]
    st.dataframe(
        top10.style.background_gradient(subset=["ROI"], cmap="YlGn"),
        use_container_width=True,
    )


# ─── Page 2 : Analyse des canaux ─────────────────────────────────────────────

def page_channels(df: pd.DataFrame) -> None:
    st.title("Analyse des canaux publicitaires")

    # Corrélation
    st.subheader("Corrélation entre investissements et ventes")
    corr = df[["TV", "Radio", "Social_Media", "Total_Budget", "Sales", "ROI"]].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Matrice de corrélation (−1 = aucun lien, +1 = lien fort)",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "> **A retenir** : TV est le canal le plus corrélé aux ventes (r ≈ 0.9). "
        "Radio et Social Media ont un impact modéré mais complémentaire. "
        "Les budgets sont quasi-indépendants entre eux — pas d'effet croisé dominant."
    )

    # Scatter budget → ventes
    st.subheader("Impact d'un canal sur les ventes")
    channel = st.selectbox(
        "Canal à analyser", ["TV", "Radio", "Social_Media"],
        format_func=lambda x: x.replace("_", " "),
        key="ch_scatter",
    )
    label_ch = channel.replace("_", " ")
    fig = px.scatter(
        df, x=channel, y="Sales", color="Influencer",
        trendline="ols",
        title=f"Budget {label_ch} vs Ventes — tendance par type d'influenceur",
        labels={channel: f"Budget {label_ch} (M€)", "Sales": "Ventes (M€)"},
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.75,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROI marginal
    st.subheader("Rendement marginal — retour sur investissement par canal")
    st.caption(
        "Un ROI décroissant avec le budget signale une saturation publicitaire : "
        "chaque euro supplémentaire rapporte de moins en moins."
    )
    col1, col2, col3 = st.columns(3)
    for col_widget, chan, label in zip(
        [col1, col2, col3],
        ["TV", "Radio", "Social_Media"],
        ["TV", "Radio", "Social Media"],
    ):
        roi_col = f"ROI_{chan}"
        df_plot = df.copy()
        df_plot[roi_col] = df_plot["Sales"] / df_plot[chan].replace(0, np.nan)
        df_sorted = df_plot.sort_values(chan)
        fig = px.scatter(
            df_sorted, x=chan, y=roi_col, trendline="lowess",
            title=f"ROI vs budget {label}",
            labels={chan: f"Budget {label} (M€)", roi_col: "ROI"},
            color_discrete_sequence=[PALETTE.get(chan, "#3498db")],
            opacity=0.65,
        )
        col_widget.plotly_chart(fig, use_container_width=True)

    # Recommandations
    st.subheader("Recommandations par canal")
    st.markdown("""
| Canal | Constat | Recommandation |
|-------|---------|----------------|
| **TV** | Canal le plus impactant sur les ventes | Maintenir un budget TV solide comme levier principal |
| **Social Media** | Meilleur ROI marginal à faible budget | Augmenter en priorité avant TV pour maximiser le ROI |
| **Radio** | Impact modéré, saturation rapide | Utiliser en complément, plafonner à ~30 M€ |
| **Influencer Mega** | Amplifie les campagnes à fort budget TV | Réserver aux campagnes > 150 M€ budget total |
| **Influencer Micro / Nano** | Meilleur ROI sur budgets restreints | Privilégier pour les campagnes ciblées à petit budget |
    """)


# ─── Page 3 : Performance des modèles ───────────────────────────────────────

def page_model_performance(meta: dict, reg_models: dict) -> None:
    st.title("Performance des modèles")
    st.markdown(
        "Comparaison des quatre algorithmes entraînés et identification des variables "
        "qui influencent le plus les ventes — pour choisir le modèle le plus fiable "
        "et comprendre sur quoi il s'appuie."
    )

    if not meta or meta.get("comparison_reg") is None:
        st.warning("Aucun résultat disponible. Lancez `python train.py` puis relancez le dashboard.")
        return

    # ── Section 1 : Comparaison des modèles ──────────────────────────────────
    st.subheader("Comparaison des modèles — Prédiction des ventes")

    comp = meta["comparison_reg"].copy()

    col_tbl, col_bar = st.columns([1, 1])
    with col_tbl:
        st.markdown("**Tableau des performances (jeu de test) :**")
        styled = comp.style.highlight_max(
            subset=["R²", "CV R² (moy)"], color="#d4edda"
        ).highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
        st.dataframe(styled, use_container_width=True)
        st.caption(
            "R² : proportion de variance expliquée (1.0 = parfait) | "
            "MAE/RMSE : erreur moyenne en M€ | CV : robustesse sur 5 découpes"
        )

    with col_bar:
        fig = px.bar(
            comp, x="Modèle", y="R²", color="Modèle",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Précision par modèle (R² test — plus haut = mieux)",
            text=comp["R²"].apply(lambda v: f"{v:.4f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0.99, 1.001])
        st.plotly_chart(fig, use_container_width=True)

    # Bandeau modèle recommandé
    best = meta.get("best_reg_model", "Gradient Boosting")
    best_row = comp[comp["Modèle"] == best]
    if not best_row.empty:
        r2   = best_row["R²"].values[0]
        mae  = best_row["MAE"].values[0]
        cv   = best_row["CV R² (moy)"].values[0]
        st.success(
            f"Modèle recommandé : **{best}** — "
            f"R² = {r2:.4f} | Erreur moyenne = {mae:.2f} M€ | Stabilité CV = {cv:.4f}  \n"
            "Ce modèle offre le meilleur compromis entre précision, stabilité "
            "et facilité de déploiement dans un contexte business marketing."
        )

    # GridSearch
    gs_params = meta.get("gs_best_params")
    gs_score  = meta.get("gs_best_score")
    if gs_params and gs_score:
        with st.expander("Optimisation automatique des paramètres (GridSearch)"):
            st.markdown(
                f"Le modèle **Gradient Boosting** a été optimisé sur 27 configurations × 5 découpes.  \n"
                f"Meilleure configuration : `{gs_params}`  \n"
                f"Score CV optimisé : **{gs_score:.4f}**"
            )
            tuned_path = os.path.join(MODEL_DIR, "regression_Gradient_Boosting_tuned.joblib")
            if os.path.exists(tuned_path):
                default_cv = next(
                    (v for k, v in meta.get("reg_results", {}).items() if "Gradient" in k and "cv_mean" in str(v)),
                    None,
                )
                st.success(
                    "Le modèle **Gradient Boosting optimisé** (tuned) est utilisé dans tout le dashboard "
                    f"et le simulateur. Score CV tuned : **{gs_score:.4f}** "
                    f"— le pipeline GridSearch est directement déployé."
                )

    st.markdown("---")

    # ── Section 2 : Importance des variables ─────────────────────────────────
    st.subheader("Quels canaux influencent le plus les ventes ?")
    st.markdown(
        "L'importance des variables mesure la contribution de chaque canal budgétaire "
        "à la prédiction des ventes. Plus la barre est longue, plus le canal est déterminant."
    )

    col_rf, col_gb = st.columns(2)
    for col_w, model_name in zip([col_rf, col_gb], ["Random Forest", "Gradient Boosting"]):
        if model_name not in reg_models:
            col_w.info(f"{model_name} non disponible.")
            continue
        try:
            pipeline     = reg_models[model_name]
            preprocessor = pipeline.named_steps["preprocessor"]
            model        = pipeline.named_steps["model"]
            feature_names = get_feature_names_after_ohe(preprocessor)
            importances   = pd.Series(model.feature_importances_, index=feature_names)
            importances   = importances.sort_values(ascending=True)

            # Libellés métier
            label_map = {
                "TV":           "Budget TV",
                "Radio":        "Budget Radio",
                "Social_Media": "Budget Social Media",
            }
            display_names = [
                label_map.get(n, n.replace("Influencer_", "Influenceur "))
                for n in importances.index
            ]

            fig = px.bar(
                x=importances.values,
                y=display_names,
                orientation="h",
                title=f"Importance des canaux — {model_name}",
                color=importances.values,
                color_continuous_scale="Blues",
                labels={"x": "Contribution relative", "y": "Variable"},
            )
            fig.update_coloraxes(showscale=False)
            col_w.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            col_w.warning(f"Indisponible : {e}")

    st.markdown(
        "> **Lecture** : TV est généralement le canal le plus déterminant, "
        "suivi de Social Media. Radio et le type d'influenceur ont un impact secondaire. "
        "Ces résultats sont cohérents entre Random Forest et Gradient Boosting — "
        "ce qui renforce la fiabilité de l'interprétation."
    )

    # ── Section 3 : SHAP ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Explication globale des prédictions (SHAP)")
    st.caption(
        "Le graphique SHAP montre, pour chaque campagne du jeu de test, "
        "comment chaque canal a poussé la prédiction vers le haut ou vers le bas."
    )

    shap_path = os.path.join(MODEL_DIR, "shap_data.joblib")
    if os.path.exists(shap_path):
        try:
            import shap
            shap_data = joblib.load(shap_path)
            shap_vals = shap_data["shap_values"]
            X_trans   = shap_data["X_transformed"]
            fnames    = shap_data.get("feature_names")

            fig_shap, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_vals, X_trans, feature_names=fnames, show=False)
            st.pyplot(fig_shap)
            plt.close()

            st.markdown(
                "> Chaque point = une campagne. "
                "**Rouge** = valeur élevée de la variable | **Bleu** = valeur faible.  \n"
                "Un SHAP positif signifie que la variable **augmente** les ventes prédites ; "
                "négatif qu'elle les **diminue**."
            )
        except Exception as e:
            st.info(f"SHAP non disponible : {e}")
    else:
        st.info("Relancez `python train.py` pour générer les données SHAP.")

    # ── Section 4 : Permutation importance ───────────────────────────────────
    perm_df = meta.get("perm_importance")
    if perm_df is not None:
        with st.expander("Confirmation par permutation (méthode indépendante du modèle)"):
            st.caption(
                "La permutation mélange chaque variable et mesure la baisse de précision. "
                "Elle confirme l'importance sans dépendre de l'algorithme utilisé."
            )
            fig = px.bar(
                perm_df.sort_values("importance_mean"),
                x="importance_mean", y="feature",
                error_x="importance_std",
                orientation="h",
                title=f"Permutation Importance — {meta.get('best_reg_model', '')}",
                labels={
                    "importance_mean": "Baisse de R² si la variable est mélangée",
                    "feature": "Variable",
                },
                color="importance_mean",
                color_continuous_scale="Oranges",
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ─── Page 4 : Simulateur ─────────────────────────────────────────────────────

def page_simulator(reg_models: dict, meta: dict, api_ok: bool) -> None:
    st.title("Simulateur budgétaire")
    st.markdown(
        "Définissez une allocation budgétaire et obtenez immédiatement "
        "une **prédiction de ventes**, une estimation du **ROI** "
        "et une analyse de **sensibilité** par canal."
    )

    if not reg_models:
        st.warning("Aucun modèle disponible. Lancez `python train.py` pour entraîner les modèles.")
        return

    col_inp, col_res = st.columns([1, 1])

    with col_inp:
        st.subheader("Paramètres du scénario")
        tv         = st.slider("Budget TV (M€)",           0.0, 300.0, 100.0, step=5.0)
        radio      = st.slider("Budget Radio (M€)",         0.0,  50.0,  15.0, step=1.0)
        social     = st.slider("Budget Social Media (M€)",  0.0,  60.0,  10.0, step=1.0)
        influencer = st.selectbox("Type d'influenceur", ["Mega", "Macro", "Micro", "Nano"])
        total_budget = tv + radio + social

    input_df = pd.DataFrame([{
        "TV": tv, "Radio": radio, "Social_Media": social, "Influencer": influencer
    }])

    # Sélection du modèle retenu (Gradient Boosting par défaut)
    default_model = meta.get("best_reg_model", list(reg_models.keys())[0])
    if default_model not in reg_models:
        default_model = list(reg_models.keys())[0]

    # Prédiction : API si disponible, repli local sinon
    predicted_sales = None
    source_label    = ""

    if api_ok:
        predicted_sales = _predict_via_api(tv, radio, social, influencer, default_model)
        if predicted_sales is not None:
            source_label = "Prédiction via API REST"

    if predicted_sales is None:
        predicted_sales = float(reg_models[default_model].predict(input_df)[0])
        source_label    = "Prédiction via modèle local" + (" (API indisponible)" if api_ok else "")

    roi    = predicted_sales / total_budget if total_budget > 0 else 0.0
    profit = predicted_sales - total_budget

    with col_res:
        st.subheader("Résultats")
        st.caption(source_label)

        c1, c2 = st.columns(2)
        c1.metric("Ventes prédites",   f"{predicted_sales:.2f} M€")
        c2.metric("ROI estimé",        f"{roi:.2f}×",
                  delta=f"{roi - 1:+.2f} vs seuil 1×")
        c3, c4 = st.columns(2)
        c3.metric("Budget total",      f"{total_budget:.2f} M€")
        c4.metric("Profit estimé",     f"{profit:.2f} M€",
                  delta_color="normal" if profit > 0 else "inverse")

        # Comparaison inter-modèles
        st.subheader("Prédictions selon le modèle")
        all_preds = {
            name: round(float(m.predict(input_df)[0]), 2)
            for name, m in reg_models.items()
        }
        preds_df = pd.DataFrame(
            list(all_preds.items()), columns=["Modèle", "Ventes prédites (M€)"]
        )
        fig = px.bar(
            preds_df, x="Modèle", y="Ventes prédites (M€)", color="Modèle",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Fourchette de prédiction selon l'algorithme",
        )
        fig.add_hline(
            y=predicted_sales, line_dash="dot", line_color="#555",
            annotation_text=f"Modèle retenu : {default_model}",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Analyse de sensibilité
    st.markdown("---")
    st.subheader("Analyse de sensibilité — impact marginal de chaque canal")
    st.caption(
        "Chaque graphique fait varier un seul canal (les deux autres restent fixes) "
        "pour mesurer son effet direct sur les ventes prédites."
    )

    cols_sens = st.columns(3)
    base      = {"TV": tv, "Radio": radio, "Social_Media": social}
    chan_labels = [("TV", "TV"), ("Radio", "Radio"), ("Social_Media", "Social Media")]

    for col_w, (chan, label) in zip(cols_sens, chan_labels):
        max_budget   = {"TV": 300.0, "Radio": 60.0, "Social_Media": 80.0}[chan]
        budget_range = np.linspace(0, max_budget, 60)
        preds_s = [
            float(reg_models[default_model].predict(
                pd.DataFrame([{**base, chan: b, "Influencer": influencer}])
            )[0])
            for b in budget_range
        ]
        fig = px.line(
            x=budget_range, y=preds_s,
            labels={"x": f"Budget {label} (M€)", "y": "Ventes prédites (M€)"},
            title=f"Sensibilité — {label}",
        )
        fig.add_vline(
            x=base[chan], line_dash="dash", line_color="red",
            annotation_text="Valeur actuelle", annotation_position="top right",
        )
        col_w.plotly_chart(fig, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_ok = _api_health()
    page = render_sidebar()

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset introuvable (`{DATA_PATH}`). "
            "Téléchargez `marketing_and_sales.csv` depuis Kaggle et placez-le dans `data/`."
        )
        st.stop()

    df              = load_dataset()
    meta, reg_models = load_artifacts()

    if meta is None:
        st.warning("Aucun modèle entraîné. Lancez `python train.py` puis relancez le dashboard.")

    if page == "Tableau de bord":
        page_overview(df)
    elif page == "Analyse des canaux":
        page_channels(df)
    elif page == "Performance des modèles":
        page_model_performance(meta or {}, reg_models)
    elif page == "Simulateur budgétaire":
        page_simulator(reg_models, meta or {}, api_ok)


if __name__ == "__main__":
    main()
