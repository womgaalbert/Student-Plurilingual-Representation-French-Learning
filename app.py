"""
app.py — Dashboard Streamlit interactif
French-Learning-Perceptions ML
Usage : streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="FLP Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Language ────────────────────────────────────────────────────────────────────

if "lang" not in st.session_state:
    st.session_state.lang = "FR"

TR = {
    "FR": {
        "title": "🌍 French-Learning-Perceptions Dashboard",
        "subtitle": "Perceptions de l'apprentissage du francais en contexte plurilingue au Cameroun",
        "nav_home": "🏠 Accueil",
        "nav_h1": "H1 — Usage quotidien",
        "nav_h2": "H2 — Motivation & Difficultés",
        "nav_h3": "H3 — Attitude envers le français",
        "nav_h4": "H4 — Intégration langues locales",
        "respondents": "Repondants",
        "establishments": "etablissements",
        "validated": "Hypotheses validees",
        "models_prod": "Modeles en prod",
        "pipeline": "Pipeline",
        "steps": "etapes",
        "mlops_level": "MLOps Level 2",
        "hypothesis_table": "4 Hypotheses",
        "tech_stack": "🛠️ Stack Technique",
        "viz": "📊 Visualisations disponibles",
        "sidebar_research": "Recherche",
        "sidebar_ml": "ML Engineering",
        "sidebar_models": "Modeles charges",
        "predict_btn": "🔮 Predire",
        "profile": "Profil de l'eleve",
        "result": "Resultat",
        "metrics_title": "Metriques",
        "threshold": "Seuil",
        "descriptive_title": "Analyse descriptive",
        "demographics_title": "Demographie — Profil des repondants",
        "yes": "Oui",
        "no": "Non",
        "male": "M",
        "female": "F",
        "languages_count": "Nombre de langues parlees",
        "age": "Age",
        "gender": "Sexe",
        "probability": "Probabilite",
        "decision_threshold": "Seuil de decision : 50%",
        "attitude_score": "Score d'attitude",
        "engagement_score": "Score d'engagement",
        "h1_desc": "Les eleves avec ≥3 langues mobilisent-ils davantage leurs langues au quotidien ?",
        "h2_desc": "La perception du francais comme 'difficile' predit-elle une motivation plus faible ?",
        "h3_desc": "L'exposition aux autres langues influence-t-elle l'attitude envers le francais ?",
        "h4_desc": "L'integration des langues camerounaises augmenterait-elle la motivation des eleves ?",
        "uses_daily": "utilise d'autres langues au quotidien",
        "no_daily": "n'utilise pas d'autres langues au quotidien",
        "motivation_label": "Motivation",
        "disciplines_wanted": "Disciplines souhaitees pour l'integration",
        "motivated": "Motivé",
        "not_motivated": "Peu motivé",
        "footer": "French-Learning-Perceptions in Plurilingual Cameroon — MLOps Level 2 — juin 2026",
    },
    "EN": {
        "title": "🌍 French-Learning-Perceptions Dashboard",
        "subtitle": "Perceptions of French learning in a plurilingual context in Cameroon",
        "nav_home": "🏠 Home",
        "nav_h1": "H1 — Daily Usage",
        "nav_h2": "H2 — Motivation & Difficulties",
        "nav_h3": "H3 — Attitude towards French",
        "nav_h4": "H4 — Local Language Integration",
        "respondents": "Respondents",
        "establishments": "schools",
        "validated": "Hypotheses validated",
        "models_prod": "Models in prod",
        "pipeline": "Pipeline",
        "steps": "steps",
        "mlops_level": "MLOps Level 2",
        "hypothesis_table": "4 Hypotheses",
        "tech_stack": "🛠️ Tech Stack",
        "viz": "📊 Available Visualizations",
        "sidebar_research": "Research",
        "sidebar_ml": "ML Engineering",
        "sidebar_models": "Loaded models",
        "predict_btn": "🔮 Predict",
        "profile": "Student Profile",
        "result": "Result",
        "metrics_title": "Metrics",
        "threshold": "Threshold",
        "descriptive_title": "Descriptive Analysis",
        "demographics_title": "Demographics — Respondent Profile",
        "yes": "Yes",
        "no": "No",
        "male": "M",
        "female": "F",
        "languages_count": "Number of languages spoken",
        "age": "Age",
        "gender": "Gender",
        "probability": "Probability",
        "decision_threshold": "Decision threshold: 50%",
        "attitude_score": "Attitude score",
        "engagement_score": "Engagement score",
        "h1_desc": "Do students with ≥3 languages use them more in daily life?",
        "h2_desc": "Does perceiving French as 'difficult' predict lower motivation?",
        "h3_desc": "Does exposure to other languages influence attitudes toward French?",
        "h4_desc": "Would integrating Cameroonian languages increase student motivation?",
        "uses_daily": "uses other languages daily",
        "no_daily": "does not use other languages daily",
        "motivation_label": "Motivation",
        "disciplines_wanted": "Preferred disciplines for integration",
        "motivated": "Motivated",
        "not_motivated": "Not motivated",
        "footer": "French-Learning-Perceptions in Plurilingual Cameroon — MLOps Level 2 — June 2026",
    },
}

def t(key: str) -> str:
    return TR[st.session_state.lang].get(key, key)

# ── Load models (cached) — cloud-compatible ────────────────────────────────────

@st.cache_resource
def load_models():
    """Charge les modeles .pkl directement (sans FastAPI) pour Streamlit Cloud."""
    import pickle as _pk
    import numpy as _np
    models, feats = {}, {}

    # Dossier models : local ou deploy_cloud (Streamlit Cloud)
    candidates = [Path("models"), Path("deploy_cloud/models")]
    model_dir = candidates[0]
    for d in candidates:
        if d.is_dir() and any(d.glob("h*/*.pkl")):
            model_dir = d
            break

    def _latest(pattern):
        return sorted(model_dir.glob(pattern),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def _load(key, pattern, path_hint=""):
        try:
            files = _latest(pattern)
            if files:
                with open(files[0], "rb") as f:
                    models[key] = _pk.load(f)
                try:
                    feats[key] = list(models[key].feature_names_in_)
                except Exception:
                    feats[key] = []
                return True
        except Exception as e:
            st.warning(f"Modèle {path_hint or key} non chargé: {e}")
        return False

    # H1
    _load("h1", "h1/*.pkl", "h1")
    # H2
    _load("h2", "h2/*A_motivation_tuned*.pkl", "h2")
    # H3
    _load("h3_reg", "h3/*reg_tuned*.pkl", "h3_reg")
    _load("h3_clf", "h3/*clf_tuned*.pkl", "h3_clf")
    # H4
    _load("h4_a", "h4/*A_motivation_tuned*.pkl", "h4_a")
    _load("h4_b", "h4/*B_engagement_tuned*.pkl", "h4_b")
    _load("h4_c", "h4/*C_discipline_tuned*.pkl", "h4_c")

    return models, feats

# ── Monkey-patch: sklearn 1.6 → 1.9 compat pour SimpleImputer ───────────
_ORIG_TRANSFORM = None

def _patch_simple_imputer():
    """Injecte _fill_dtype sur tous les SimpleImputer avant transform."""
    global _ORIG_TRANSFORM
    from sklearn.impute import SimpleImputer
    if _ORIG_TRANSFORM is not None:
        return  # déjà patché
    _ORIG_TRANSFORM = SimpleImputer.transform
    def _patched_transform(self, X):
        if not hasattr(self, "_fill_dtype") or self._fill_dtype is None:
            import numpy as _np
            stats = getattr(self, "statistics_", None)
            if stats is not None and hasattr(stats, "flat"):
                vals = [s for s in stats.flat if not _np.isnan(s)]
                self._fill_dtype = _np.result_type(*vals) if vals else _np.float64
            else:
                self._fill_dtype = _np.float64
        return _ORIG_TRANSFORM(self, X)
    SimpleImputer.transform = _patched_transform

_patch_simple_imputer()

try:
    MODELS, FEATURE_COLS = load_models()
except Exception as _e:
    st.error(f"Erreur chargement modèles: {_e}")
    MODELS, FEATURE_COLS = {}, {}

if not MODELS:
    st.warning("Aucun modèle chargé. L'app fonctionne en mode démo.")

# ── Reports path helper ──────────────────────────────────────────────────────

def _report_dir(subpath: str = "") -> Path:
    """Retourne le dossier reports (local ou deploy_cloud/reports en fallback)."""
    for root in (Path("reports"), Path("deploy_cloud/reports")):
        p = root / subpath
        if p.exists():
            return p
    return Path("reports") / subpath

# ── Constants ─────────────────────────────────────────────────────────────────

DESCRIPTIVE_DIRS = {
    "H1": _report_dir("h1/descriptive"),
    "H2": _report_dir("h2/descriptive"),
    "H3": _report_dir("h3/descriptive"),
    "H4": _report_dir("h4/descriptive"),
}

INTERPRETATIONS = {
    "wordcloud_global.png": (
        "**Nuage de mots global** — Vocabulaire dominant dans les reponses textuelles. "
        "Les mots les plus gros sont les plus frequents. Observer les themes principaux "
        "qui emergent spontanement du discours des eleves."
    ),
    "wordcloud_representation_affective.png": (
        "**Representation affective** — Termes lies aux emotions (aimer, peur, difficile, plaisir). "
        "Cette carte revele le rapport emotionnel des eleves au francais : est-il positif, "
        "anxieux, ou ambivalent ? Dominante ici : la perception du francais comme 'difficile' "
        "coexiste avec un attachement certain."
    ),
    "wordcloud_representation_identitaire.png": (
        "**Representation identitaire** — Marqueurs d'appartenance linguistique et culturelle. "
        "Les eleves se positionnent-ils comme 'locuteurs natifs', 'etrangers' au francais, "
        "ou 'bilingues' ? La presence des noms de langues nationales (ewondo, foufoulde) "
        "montre un ancrage identitaire fort dans le plurilinguisme local."
    ),
    "wordcloud_representation_institutionnelle.png": (
        "**Representation institutionnelle** — Vocabulaire lie a l'ecole, au professeur, "
        "a l'obligation scolaire. Cette carte montre comment les eleves percoivent le francais "
        "comme langue de l'institution : matiere scolaire imposee ou opportunite d'ascension ?"
    ),
    "wordcloud_representation_utilitaire.png": (
        "**Representation utilitaire** — Termes lies a la necessite, la reussite, le travail, "
        "l'avenir professionnel. Cette carte revele la perception instrumentale du francais : "
        "langue-outil pour les etudes, l'emploi, la mobilite sociale."
    ),
    "wordcloud_resistance_contrainte.png": (
        "**Resistance / contrainte** — Termes evoquant la contrainte (force, oblige, "
        "pas le choix). Cette carte est cruciale : elle capture les discours de resistance "
        "ou de resignation face a l'imposition du francais. Une presence significative ici "
        "signale un malaise linguistique a prendre au serieux."
    ),
    "umap_clusters.png": (
        "**UMAP — Clusters K-Means (n=5)** — Projection 2D des embeddings CamemBERT. "
        "Chaque point = un repondant. Les couleurs = clusters semantiques. "
        "Points proches = discours similaires. Points isoles = profils atypiques. "
        "Une concentration dense indique un discours homogene ; une dispersion indique "
        "une diversite des points de vue. C'est la carte la plus informative pour "
        "comprendre la structure globale du discours des eleves."
    ),
    "umap_apriori.png": (
        "**UMAP — Categories A Priori** — Meme projection que ci-dessus, mais coloree "
        "par la categorie A Priori dominante (cadre Moscovici & Jodelet). "
        "Permet de verifier si les categories theoriques forment des ilots coherents "
        "dans l'espace semantique, ou si elles se chevauchent."
    ),
    "umap_stereotypes.png": (
        "**UMAP — Stereotypes detectes** — Projection coloree par les stereotypes "
        "et preconstruits identifies par similarite cosinus avec des marqueurs cibles. "
        "Permet de localiser les discours porteurs de prejuges linguistiques et d'evaluer "
        "leur ampleur dans l'echantillon."
    ),
    "apriori_distribution.png": (
        "**Distribution A Priori** — Effectifs par categorie de representation "
        "(identitaire, affective, institutionnelle, utilitaire, resistance/contrainte). "
        "La categorie dominante indique le prisme principal a travers lequel les eleves "
        "percoivent le francais. Une dominance identitaire + resistance signale une "
        "tension entre langue locale et langue scolaire."
    ),
    "stereotype_distribution.png": (
        "**Distribution des stereotypes** — Nombre d'eleves par type de preconstruit "
        "detecte (distanciation identitaire, preconstruit de difficulte, auto-devalorisation, "
        "exclusion symbolique). Un score eleve sur 'auto-devalorisation' ou 'exclusion' "
        "est un signal d'alerte pedagogique fort."
    ),
    "heatmap_cooccurrence.png": (
        "**Heatmap de co-occurrence** — Intensite des associations entre termes cles. "
        "Les cases rouges indiquent des paires de mots qui apparaissent souvent ensemble "
        "dans les reponses. La diagonale montre les termes les plus frequents. "
        "Observer les blocs de forte correlation : ils revelent les champs semantiques "
        "qui structurent le discours (ex: 'grammaire+conjugaison+verbes')."
    ),
    "network_cooccurrence.png": (
        "**Reseau de co-occurrences (NetworkX)** — Graphe des associations entre termes. "
        "Les hubs (gros nœuds) sont les mots-cles autour desquels s'organise le discours. "
        "Les aretes (liens) representent les co-occurrences frequentes. "
        "Une structure en etoile autour de 'oui'/'non' indique un discours polarise ; "
        "une structure en communautes indique des themes distincts."
    ),
    "ngrams_unigrammes.png": (
        "**Top-20 unigrammes** — Mots isoles les plus frequents. Permet d'identifier "
        "le vocabulaire dominant. La presence massive de 'oui' indique des reponses "
        "affirmatives ; les termes comme 'grammaire', 'langue', 'communiquer' revelent "
        "les preoccupations principales des eleves."
    ),
    "ngrams_bigrammes.png": (
        "**Top-20 bigrammes** — Paires de mots consecutifs les plus frequentes. "
        "Les bigrammes revelent les expressions recurrentes : 'tres difficile', "
        "'langue maternelle', 'parce que'. Ils capturent mieux le sens que les "
        "unigrammes car ils preservent le contexte immediat."
    ),
    "ngrams_trigrammes.png": (
        "**Top-20 trigrammes** — Triplets de mots consecutifs. Ils capturent des "
        "formules figees et des expressions completes : 'parler plusieurs langues', "
        "'apprendre le francais', 'c'est pas facile'. Les trigrammes sont les plus "
        "proches du discours naturel des eleves."
    ),
}

CAMEMBERT_TEXT = """
**CamemBERT (camembert-base)** : modele de langue francais pre-entraine par
INRIA/Facebook. Les embeddings (768 dimensions) sont reduits par PCA (20D,
variance expliquee 84-89%) et utilises comme features ML pour H3 et H4.
Les projections UMAP visualisent les clusters semantiques et les categories
A Priori dans l'espace des embeddings.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict(model, df: pd.DataFrame, model_key: str) -> np.ndarray:
    """Pad df with zeros for missing training columns, then predict."""
    if not model:
        return np.array([])
    expected = FEATURE_COLS.get(model_key, [])
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    order = [c for c in expected if c in df.columns]
    return model.predict(df[order])


def predict_proba(model, df: pd.DataFrame, model_key: str) -> np.ndarray:
    if not model:
        return np.array([[]])
    expected = FEATURE_COLS.get(model_key, [])
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    order = [c for c in expected if c in df.columns]
    return model.predict_proba(df[order])


def metric_card(title, value, delta=None, color=None):
    c = f"<span style='color:{color};font-weight:bold;font-size:1.4em'>{value}</span>" if color else f"<span style='font-weight:bold;font-size:1.4em'>{value}</span>"
    st.metric(title, value, delta=delta)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🌍 FLP Dashboard")

lang = st.sidebar.selectbox("🌐 Langue / Language", ["FR", "EN"],
                             index=0 if st.session_state.lang == "FR" else 1,
                             key="lang_selector",
                             on_change=lambda: st.session_state.update(
                                 {"lang": st.session_state.lang_selector}))
st.session_state.lang = lang

st.sidebar.caption("French-Learning-Perceptions in Plurilingual Cameroon")
st.sidebar.markdown("---")

hypothesis = st.sidebar.radio(
    "Hypothese" if lang == "FR" else "Hypothesis",
    [t("nav_home"), t("nav_h1"), t("nav_h2"), t("nav_h3"), t("nav_h4")],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('sidebar_research')}**")
st.sidebar.markdown("Chancelline Armelle Nongni Kendjio")
st.sidebar.markdown("Universite Marie et Louis Pasteur de Besancon (France)")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('sidebar_ml')}**")
st.sidebar.markdown("Albert Womga")
st.sidebar.markdown("---")
st.sidebar.caption(f"{t('sidebar_models')} : {len(MODELS)}")
for k in MODELS:
    st.sidebar.caption(f"  • {k}")

# ── Home ──────────────────────────────────────────────────────────────────────

if hypothesis == t("nav_home"):
    st.title(t("title"))
    st.markdown(f"### {t('subtitle')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t("respondents"), "500", f"15 {t('establishments')}")
    with col2:
        st.metric(t("validated"), "3/4", "H1, H2, H4")
    with col3:
        st.metric(t("models_prod"), "7", "Model Registry")
    with col4:
        st.metric(t("pipeline"), f"10 {t('steps')}", t("mlops_level"))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🎯 {t('hypothesis_table')}")
        if lang == "FR":
            st.markdown("""
            | H | Hypothese | Modele | Statut |
            |---|-----------|--------|--------|
            | H1 | Repertoire → usage quotidien | XGBoost | ✅ F1=0.83 |
            | H2 | Perception → motivation | ClassChain+XGB | ✅ F1=0.95 |
            | H3 | Exposition → attitude | VotingRegressor | ⚠️ MAE=0.51 |
            | H4 | Langues locales → engagement | XGBoost+CamemBERT | ✅ rho=0.56 |
            """)
        else:
            st.markdown("""
            | H | Hypothesis | Model | Status |
            |---|-----------|-------|--------|
            | H1 | Repertoire → daily usage | XGBoost | ✅ F1=0.83 |
            | H2 | Perception → motivation | ClassChain+XGB | ✅ F1=0.95 |
            | H3 | Exposure → attitude | VotingRegressor | ⚠️ MAE=0.51 |
            | H4 | Local languages → engagement | XGBoost+CamemBERT | ✅ rho=0.56 |
            """)

    with col2:
        st.markdown(t("tech_stack"))
        if lang == "FR":
            st.markdown("""
            - **NLP** : CamemBERT (embeddings PCA 20D)
            - **ML** : XGBoost, ExtraTrees, VotingRegressor, ClassifierChain
            - **MLOps** : MLflow (116 runs), FastAPI, Docker, Model Registry
            - **Tests** : 42/42 ✅ pytest
            - **Rapports** : Word + JSON generation automatique
            """)
        else:
            st.markdown("""
            - **NLP**: CamemBERT (PCA 20D embeddings)
            - **ML**: XGBoost, ExtraTrees, VotingRegressor, ClassifierChain
            - **MLOps**: MLflow (116 runs), FastAPI, Docker, Model Registry
            - **Tests**: 42/42 ✅ pytest
            - **Reports**: Automated Word + JSON generation
            """)

    st.markdown("---")
    st.markdown(f"### {t('viz')}")

    viz_col1, viz_col2, viz_col3 = st.columns(3)
    reports = _report_dir()

    # Wordcloud
    wc = reports / "h1" / "descriptive" / "wordcloud_global.png"
    if wc.exists():
        with viz_col1:
            st.image(str(wc), caption="Nuage de mots — H1", use_container_width=True)

    # UMAP
    umap = reports / "h2" / "descriptive" / "umap_clusters.png"
    if umap.exists():
        with viz_col2:
            st.image(str(umap), caption="UMAP Clusters — H2", use_container_width=True)

    # Network
    net = reports / "h1" / "descriptive" / "network_cooccurrence.png"
    if net.exists():
        with viz_col3:
            st.image(str(net), caption="Reseau co-occurrences — H1", use_container_width=True)

    # Demographics
    with st.expander(f"📊 {t('demographics_title')}", expanded=False):
        demo1 = reports / "demographics" / "demographics_overview.png"
        demo2 = reports / "demographics" / "demographics_langues.png"
        dcol1, dcol2 = st.columns(2)
        if demo1.exists():
            with dcol1:
                st.image(str(demo1), caption="Profil demographique (age, classe, region, genre, langue maternelle)", use_container_width=True)
        if demo2.exists():
            with dcol2:
                st.image(str(demo2), caption="Langues parlees — Top 20 (francais, anglais, ewondo, foufoulde...)", use_container_width=True)


# ── H1 ────────────────────────────────────────────────────────────────────────

elif hypothesis == t("nav_h1"):
    st.title("H1 — " + ("Repertoire multilingue & mobilisation des langues" if lang == "FR" else "Multilingual Repertoire & Language Use"))
    st.markdown(f"*{t('h1_desc')}*")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"### {t('profile')}")
        nb_langues = st.slider(t("languages_count"), 1, 10, 3)
        if lang == "FR":
            apprent = st.selectbox("A appris d'autres langues avant le francais ?", ["Oui", "Non"])
            relation = st.selectbox("Relation avec la langue maternelle", ["Aisee", "Complexe"])
            freq = st.slider("Frequence d'exposition aux autres langues", 0, 4, 3,
                             format="%d", help="0=Jamais, 1=Rarement, 2=Parfois, 3=Souvent, 4=Toujours")
            sentiment = st.slider("Sentiment envers le plurilinguisme", -1.0, 1.0, 0.5, 0.1,
                                  help="Negatif (-1) a Positif (1)")
        else:
            apprent = st.selectbox("Learned other languages before French?", ["Yes", "No"])
            relation = st.selectbox("Relationship with mother tongue", ["Fluent", "Complex"])
            freq = st.slider("Exposure frequency to other languages", 0, 4, 3,
                             format="%d", help="0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Always")
            sentiment = st.slider("Sentiment toward plurilingualism", -1.0, 1.0, 0.5, 0.1,
                                  help="Negative (-1) to Positive (1)")
        age = st.slider(t("age"), 8, 20, 13)
        sexe = st.selectbox(t("gender"), [t("female"), t("male")])

        if st.button(f"{t('predict_btn')} — H1", type="primary", use_container_width=True, disabled="h1" not in MODELS):
            df = pd.DataFrame([{
                "nb_langues": nb_langues,
                "apprent_anterieur_bin": 1 if apprent in ("Oui", "Yes") else 0,
                "relation_lm_ord": 2 if relation in ("Aisee", "Fluent") else 1,
                "domaine_usage_freq": freq,
                "valorisation_sent": sentiment,
                "sexe_bin": 1 if sexe in (t("male"), "M") else 0,
                "age": float(age),
            }])
            if "h1" not in MODELS: st.error("Modèle H1 non disponible"); st.stop()
            proba = predict_proba(MODELS["h1"], df, "h1")[0, 1]
            pred = t("yes") if proba >= 0.5 else t("no")

            with col2:
                st.markdown(f"### {t('result')}")
                if proba >= 0.5:
                    st.success(f"### {pred} — {t('uses_daily')}")
                else:
                    st.warning(f"### {pred} — {t('no_daily')}")
                st.metric(t("probability"), f"{proba:.1%}")
                st.progress(float(proba))
                st.caption(t("decision_threshold"))

    with col2:
        if "MODELS" in locals() and "h1" not in st.session_state:
            st.markdown(f"### {t('metrics_title')} H1")
            st.markdown("""
            | Metrique | Valeur | Seuil |
            |----------|--------|-------|
            | F1-macro | 0.835 | ≥0.70 |
            | ROC-AUC | 0.851 | ≥0.75 |
            | CV F1 | 0.807±0.039 | — |
            """)

    # Descriptive H1
    with st.expander(f"📊 {t('descriptive_title')} H1 — CamemBERT + Visualisations", expanded=True):
        st.markdown(CAMEMBERT_TEXT)
        d = _report_dir("h1/descriptive")
        imgs = sorted(d.glob("*.png"), key=lambda p: p.name) if d.exists() else []
        if imgs:
            for i, img in enumerate(imgs):
                if i % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if i % 2 == 0 else c2):
                    st.image(str(img), use_container_width=True)
                    st.caption(INTERPRETATIONS.get(img.name, img.name))
        else:
            st.info("Images non disponibles — lancer le pipeline descriptif")


# ── H2 ────────────────────────────────────────────────────────────────────────

elif hypothesis == t("nav_h2"):
    if lang == "FR":
        st.title("H2 — Representations du francais → Motivation & Difficultes")
        st.markdown("*La perception du francais comme 'difficile' predit-elle une motivation plus faible ?*")
    else:
        st.title("H2 — French Representations → Motivation & Difficulties")
        st.markdown("*Does perceiving French as 'difficult' predict lower motivation?*")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"### {'Perception du francais' if lang == 'FR' else 'Perception of French'}")
        st.caption("Cochez les mots qui decrivent votre perception :" if lang == "FR" else "Check words that describe your perception:")
        perc_cols = {}
        FR_LABELS = [("utile", "Utile"), ("belle", "Belle"), ("dificil", "Difficile"),
                     ("importan", "Importante"), ("compliqu", "Compliquee"),
                     ("interess", "Interessante"), ("ennuyeus", "Ennuyeuse")]
        EN_LABELS = [("utile", "Useful"), ("belle", "Beautiful"), ("dificil", "Difficult"),
                     ("importan", "Important"), ("compliqu", "Complicated"),
                     ("interess", "Interesting"), ("ennuyeus", "Boring")]
        labels_list = FR_LABELS if lang == "FR" else EN_LABELS
        for lbl, disp in labels_list:
            perc_cols[f"perc_{lbl}"] = 1 if st.checkbox(disp, value=(lbl in ["utile", "importan"])) else 0

        sentiment = st.slider("Sentiment general" if lang == "FR" else "General sentiment", -1.0, 1.0, 0.0, 0.1)
        importance = st.selectbox("Le francais est-il important ?" if lang == "FR" else "Is French important?", ["Oui", "Non"] if lang == "FR" else ["Yes", "No"])
        if lang == "FR":
            hierarchie = st.selectbox("Compare aux autres langues, le francais est...",
                                      ["Plus important", "Autant important", "Moins important"])
        else:
            hierarchie = st.selectbox("Compared to other languages, French is...",
                                      ["More important", "Equally important", "Less important"])
        age = st.slider(t("age"), 8, 20, 13)
        sexe = st.selectbox(t("gender"), [t("female"), t("male")])

        if st.button(f"{t('predict_btn')} — H2", type="primary", use_container_width=True, disabled="h2" not in MODELS):
            hier_map = {"Plus important": 3, "Autant important": 2, "Moins important": 1,
                        "More important": 3, "Equally important": 2, "Less important": 1}
            df = pd.DataFrame([{
                **perc_cols,
                "mots_assoc_sent": sentiment,
                "importance_bin": 1 if importance in ("Oui", "Yes") else 0,
                "importance_sent": sentiment,
                "hierarchie_fr": hier_map[hierarchie],
                "sexe_bin": 1 if sexe in (t("male"), "M") else 0,
                "age": float(age),
            }])
            if "h2" not in MODELS: st.error("Modèle H2 non disponible"); st.stop()
            pred = int(np.asarray(predict(MODELS["h2"], df, "h2")).flat[0])
            if lang == "FR":
                labels = {0: "Faible", 1: "Moyenne", 2: "Elevee"}
            else:
                labels = {0: "Low", 1: "Medium", 2: "High"}

            with col2:
                st.markdown(f"### {t('result')}")
                color_map = {0: "red", 1: "orange", 2: "green"}
                c = color_map.get(pred, "black")
                st.markdown(f"### {t('motivation_label')} : <span style='color:{c}'>{labels.get(pred, str(pred))}</span>",
                            unsafe_allow_html=True)

    with col2:
        st.markdown(f"### {t('metrics_title')} H2")
        st.markdown("""
        | Metrique | Valeur | Seuil |
        |----------|--------|-------|
        | F1-weighted A | 0.954 | ≥0.65 |
        | F1-micro B | 0.745 | ≥0.72 |
        | Val-F1 A | 0.98 | — |
        """)

    with st.expander("📊 Analyse descriptive H2 — CamemBERT + Stereotypes", expanded=True):
        st.markdown(CAMEMBERT_TEXT)
        d = _report_dir("h2/descriptive")
        imgs = sorted(d.glob("*.png"), key=lambda p: p.name) if d.exists() else []
        if imgs:
            for i, img in enumerate(imgs):
                if i % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if i % 2 == 0 else c2):
                    st.image(str(img), use_container_width=True)
                    st.caption(INTERPRETATIONS.get(img.name, img.name))
        else:
            st.info("Images non disponibles")


# ── H3 ────────────────────────────────────────────────────────────────────────

elif hypothesis == t("nav_h3"):
    if lang == "FR":
        st.title("H3 — Exposition plurilingue → Attitudes envers le francais")
        st.markdown("*L'exposition aux autres langues influence-t-elle l'attitude envers le francais ?*")
    else:
        st.title("H3 — Plurilingual Exposure → Attitudes toward French")
        st.markdown("*Does exposure to other languages influence attitudes toward French?*")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"### {t('profile')}")
        exposition = st.selectbox("Expose a d'autres langues ?" if lang == "FR" else "Exposed to other languages?",
                                  ["Oui", "Non"] if lang == "FR" else ["Yes", "No"])
        interet = st.selectbox("Interet pour les autres langues ?" if lang == "FR" else "Interest in other languages?",
                               ["Oui", "Non"] if lang == "FR" else ["Yes", "No"])
        interet_sent = st.slider("Sentiment — interet autres langues" if lang == "FR" else "Sentiment — interest in other languages",
                                 -1.0, 1.0, 0.3, 0.1)
        perception_sent = st.slider("Sentiment — perception plurilinguisme" if lang == "FR" else "Sentiment — perception of plurilingualism",
                                    -1.0, 1.0, 0.5, 0.1)
        if lang == "FR":
            perception_ord = st.select_slider("Perception du plurilinguisme",
                                              ["Pas du tout bien", "Un peu bien", "Plutot bien", "Tres bien"],
                                              value="Plutot bien")
            ord_map = {"Pas du tout bien": 1, "Un peu bien": 2, "Plutot bien": 3, "Tres bien": 4}
        else:
            perception_ord = st.select_slider("Perception of plurilingualism",
                                              ["Not good at all", "A little good", "Quite good", "Very good"],
                                              value="Quite good")
            ord_map = {"Not good at all": 1, "A little good": 2, "Quite good": 3, "Very good": 4}
        nb_langues = st.slider(t("languages_count"), 1, 10, 3)
        age = st.slider(t("age"), 8, 20, 13)
        sexe = st.selectbox(t("gender"), [t("female"), t("male")])

        if st.button(f"{t('predict_btn')} — H3", type="primary", use_container_width=True, disabled="h3_reg" not in MODELS):
            df = pd.DataFrame([{
                "exposition_bin": 1 if exposition in ("Oui", "Yes") else 0,
                "interet_bin": 1 if interet in ("Oui", "Yes") else 0,
                "interet_sent": interet_sent,
                "perception_multi_sent": perception_sent,
                "perception_multi_ord": ord_map[perception_ord],
                "nb_langues": nb_langues,
                "sexe_bin": 1 if sexe in (t("male"), "M") else 0,
                "age": float(age),
            }])

            if "h3_reg" not in MODELS: st.error("Modèle H3 non disponible"); st.stop()
            score = float(np.asarray(predict(MODELS["h3_reg"], df, "h3_reg")).flat[0])
            classe = int(np.asarray(predict(MODELS["h3_clf"], df, "h3_clf")).flat[0])
            labels = {0: "Negative", 1: ("Neutre" if lang == "FR" else "Neutral"), 2: "Positive"}

            with col2:
                st.markdown(f"### {t('result')}")
                st.metric(t("attitude_score"), f"{score:.2f} / 5.0")
                st.progress(min(max(score / 5.0, 0.0), 1.0))
                classe_label = labels.get(classe, str(classe))
                if classe_label == "Positive":
                    st.success(f"{'Classe' if lang == 'FR' else 'Class'} : {classe_label}")
                elif "Neut" in classe_label:
                    st.info(f"{'Classe' if lang == 'FR' else 'Class'} : {classe_label}")
                else:
                    st.warning(f"{'Classe' if lang == 'FR' else 'Class'} : {classe_label}")

    with col2:
        st.markdown(f"### {t('metrics_title')} H3")
        st.markdown("""
        | Metrique | Valeur | Seuil |
        |----------|--------|-------|
        | MAE | 0.513 | ≤0.50 |
        | F1-weighted | 0.780 | ≥0.68 |
        | Pearson p | 0.984 | <0.05 |
        """)
        if lang == "FR":
            st.info("93.5% des eleves sont exposes aux autres langues → "
                    "variance quasi-nulle pour le test causal. "
                    "MAE plafonnee par le proxy composite.")
        else:
            st.info("93.5% of students are exposed to other languages → "
                    "near-zero variance for the causal test. "
                    "MAE limited by the composite proxy.")

    with st.expander("📊 Analyse descriptive H3 — CamemBERT + Score attitude", expanded=True):
        st.markdown(CAMEMBERT_TEXT)
        st.markdown("**CamemBERT PCA 20D** integre comme features (variance expliquee 84.8%).")
        d = _report_dir("h3/descriptive")
        imgs = sorted(d.glob("*.png"), key=lambda p: p.name) if d.exists() else []
        if imgs:
            for i, img in enumerate(imgs):
                if i % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if i % 2 == 0 else c2):
                    st.image(str(img), use_container_width=True)
                    st.caption(INTERPRETATIONS.get(img.name, img.name))
        else:
            st.info("Images non disponibles")


# ── H4 ────────────────────────────────────────────────────────────────────────

elif hypothesis == t("nav_h4"):
    if lang == "FR":
        st.title("H4 — Integration des langues locales → Engagement")
        st.markdown("*L'integration des langues camerounaises augmenterait-elle la motivation des eleves ?*")
    else:
        st.title("H4 — Local Language Integration → Engagement")
        st.markdown("*Would integrating Cameroonian languages increase student motivation?*")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"### {t('profile')}")
        if lang == "FR":
            interet_camarades = st.selectbox("Interet pour les langues des camarades",
                                             ["Pas du tout", "Un peu", "Bien", "Tres bien"], index=2)
            souhait = st.select_slider("Souhait d'inclure les langues locales dans les cours",
                                       ["Jamais", "Rarement", "Parfois", "Souvent", "Toujours"], value="Souvent")
            interet_map = {"Pas du tout": 0, "Un peu": 1, "Bien": 2, "Tres bien": 3}
            freq_map = {"Jamais": 0, "Rarement": 1, "Parfois": 2, "Souvent": 3, "Toujours": 4}
        else:
            interet_camarades = st.selectbox("Interest in classmates' languages",
                                             ["Not at all", "A little", "Well", "Very well"], index=2)
            souhait = st.select_slider("Wish to include local languages in class",
                                       ["Never", "Rarely", "Sometimes", "Often", "Always"], value="Often")
            interet_map = {"Not at all": 0, "A little": 1, "Well": 2, "Very well": 3}
            freq_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
        interet_sent = st.slider("Sentiment — interet langues camarades" if lang == "FR" else "Sentiment — interest in classmates' languages",
                                 -1.0, 1.0, 0.4, 0.1)
        age = st.slider(t("age"), 8, 20, 12)
        sexe = st.selectbox(t("gender"), [t("female"), t("male")])

        if st.button(f"{t('predict_btn')} — H4", type="primary", use_container_width=True, disabled="h4_a" not in MODELS):
            df = pd.DataFrame([{
                "interet_camarades_ord": interet_map[interet_camarades],
                "interet_camarades_sent": interet_sent,
                "souhait_freq": freq_map[souhait],
                "sexe_bin": 1 if sexe in (t("male"), "M") else 0,
                "age": float(age),
            }])

            if "h4_a" not in MODELS: st.error("Modèle H4 non disponible"); st.stop()
            proba_motiv = predict_proba(MODELS["h4_a"], df, "h4_a")[0, 1]
            pred_motiv = t("motivated") if proba_motiv >= 0.5 else t("not_motivated")

            engagement = int(np.asarray(predict(MODELS["h4_b"], df, "h4_b")).flat[0] + 1)

            DISC = ["vocabulaire", "grammaire", "lecture", "expression_orale", "conjugaison"]
            disc_pred = predict(MODELS["h4_c"], df, "h4_c")[0]
            disciplines = [DISC[i] for i, v in enumerate(disc_pred) if v == 1]

            with col2:
                st.markdown(f"### {t('result')}")
                if proba_motiv >= 0.5:
                    st.success(f"### {pred_motiv}" + (" par les langues locales" if lang == "FR" else " by local languages"))
                else:
                    st.warning(f"### {pred_motiv}" + (" par les langues locales" if lang == "FR" else " by local languages"))
                st.metric(t("probability"), f"{proba_motiv:.1%}")

                st.markdown("---")
                st.metric(t("engagement_score"), f"{engagement} / 4")
                st.progress(engagement / 4.0)

                st.markdown("---")
                st.markdown(f"**{t('disciplines_wanted')} :**")
                if disciplines:
                    for d in disciplines:
                        st.markdown(f"  • {d.capitalize()}")
                else:
                    st.caption("Aucune discipline specifique detectee" if lang == "FR" else "No specific discipline detected")

    with col2:
        st.markdown(f"### {t('metrics_title')} H4")
        st.markdown("""
        | Metrique | Valeur | Seuil |
        |----------|--------|-------|
        | F1-A (motivation) | 0.807 | ≥0.70 |
        | Spearman rho | 0.561 | ≥0.55 |
        | Subset accuracy | 1.000 | ≥0.45 |
        """)
        st.success(("H4 validee grace a CamemBERT PCA 20D (+0.085 rho)" if lang == "FR" else "H4 validated thanks to CamemBERT PCA 20D (+0.085 rho)"))

    with st.expander("📊 Analyse descriptive H4 — CamemBERT + Engagement", expanded=True):
        st.markdown(CAMEMBERT_TEXT)
        st.markdown("**CamemBERT PCA 20D** (variance expliquee 88.6%). Gain rho: 0.476→0.561 (+0.085).")
        d = _report_dir("h4/descriptive")
        imgs = sorted(d.glob("*.png"), key=lambda p: p.name) if d.exists() else []
        if imgs:
            for i, img in enumerate(imgs):
                if i % 2 == 0:
                    c1, c2 = st.columns(2)
                with (c1 if i % 2 == 0 else c2):
                    st.image(str(img), use_container_width=True)
                    st.caption(INTERPRETATIONS.get(img.name, img.name))
        else:
            st.info("Images non disponibles")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(t("footer"))
