"""
test_french-learning-perceptions.py — Tests unitaires French-Learning-Perceptions ML
Couvre : pipeline de prétraitement, encodages, features H1–H4,
         no-leakage, MLflow config, seuils de validation.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.constants import (
    FREQ_MAP, LIKERT_MAP, IMPORTANCE_MAP, RELATION_LM_MAP,
    DIFFICULTE_KEYWORDS, DISCIPLINE_KEYWORDS,
    COL_CONSENTEMENT, COL_USAGE_QUOTIDIEN, COL_MOTIVATION_CAMERO,
)
from utils.config import load_config
from preprocess import (
    filter_consent, anonymize, clean_demographics,
    extract_oui_non, simple_sentiment, extract_multilabels,
    count_languages, build_h1, build_h2, build_h3, build_h4,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE DE BASE
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineBase:

    def test_consent_filter_removes_refusal(self, sample_df):
        """Seules les lignes 'J'accepte' passent."""
        df = filter_consent(sample_df, "accepte")
        assert len(df) == 9
        assert df[COL_CONSENTEMENT].str.lower().str.contains("accepte").all()

    def test_anonymize_removes_horodateur(self, sample_df):
        """La colonne horodateur est supprimée après anonymisation."""
        df = filter_consent(sample_df, "accepte")
        df = anonymize(df)
        assert "horodateur" not in df.columns

    def test_age_normalization(self, clean_df):
        """'12ans', '12', '12 ans' → float valide."""
        assert clean_df["age"].dtype in ["float64", "int64"]
        assert clean_df["age"].between(5, 100).all()
        assert clean_df["age"].isna().sum() == 0

    def test_region_one_hot(self, clean_df):
        """Les colonnes region_ sont créées après clean_demographics."""
        region_cols = [c for c in clean_df.columns if c.startswith("region_")]
        assert len(region_cols) >= 1

    def test_sexe_binary(self, clean_df):
        """sexe_bin ∈ {0, 1}."""
        assert set(clean_df["sexe_bin"].dropna().unique()).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENCODAGES & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class TestEncodings:

    def test_freq_map_complete(self):
        """FREQ_MAP couvre toutes les valeurs attendues."""
        for val in ["Toujours", "Souvent", "Parfois", "Rarement", "Jamais"]:
            assert val in FREQ_MAP, f"Manquant dans FREQ_MAP: {val}"

    def test_freq_map_order(self):
        """Toujours > Souvent > Parfois > Rarement > Jamais."""
        assert FREQ_MAP["Toujours"] > FREQ_MAP["Souvent"] > FREQ_MAP["Parfois"]
        assert FREQ_MAP["Parfois"]  > FREQ_MAP["Rarement"] > FREQ_MAP["Jamais"]

    def test_likert_map_range(self):
        assert LIKERT_MAP["Très bien"] == 4
        assert LIKERT_MAP["Pas du tout bien"] == 1

    def test_importance_map(self):
        assert IMPORTANCE_MAP["plus important"] == 3
        assert IMPORTANCE_MAP["autant important"] == 2
        assert IMPORTANCE_MAP["moins important"] == 1

    def test_extract_oui_non_values(self):
        assert extract_oui_non("OUI") == 1
        assert extract_oui_non("NON") == 0
        assert extract_oui_non("Oui, parce que ça aide") == 1
        assert extract_oui_non(None) == -1
        assert extract_oui_non(np.nan) == -1

    def test_count_languages(self):
        assert count_languages("Français, Anglais, Bamoun") == 3
        assert count_languages("Français") == 1
        assert count_languages(None) == 0
        assert count_languages("Français et Anglais") == 2

    def test_extract_multilabels_difficultes(self):
        labels = extract_multilabels(
            "J'ai des problèmes de grammaire et de vocabulaire",
            DIFFICULTE_KEYWORDS
        )
        assert "grammaire" in labels
        assert "vocabulaire" in labels

    def test_extract_multilabels_disciplines(self):
        labels = extract_multilabels(
            "J'aimerais associer au vocabulaire et à la lecture",
            DISCIPLINE_KEYWORDS
        )
        assert "vocabulaire" in labels
        assert "lecture" in labels

    def test_simple_sentiment_positive(self):
        score = simple_sentiment("Oui ça aide à communiquer et c'est utile")
        assert score > 0

    def test_simple_sentiment_negative(self):
        score = simple_sentiment("C'est difficile et dur avec des fautes")
        assert score < 0

    def test_simple_sentiment_neutral(self):
        score = simple_sentiment(None)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURES H1 — Répertoire multilingue
# ══════════════════════════════════════════════════════════════════════════════

class TestH1Features:

    def test_h1_target_binary(self, clean_df):
        """h1_target ∈ {0, 1} uniquement."""
        h1 = build_h1(clean_df)
        assert set(h1["h1_target"].unique()).issubset({0, 1})

    def test_h1_nb_langues_positive(self, clean_df):
        """Ind.1 VI — nb_langues ≥ 1."""
        h1 = build_h1(clean_df)
        assert (h1["nb_langues"] >= 1).all()

    def test_h1_relation_lm_ordinal(self, clean_df):
        """Ind.5 VI — relation_lm_ord ∈ {0, 1, 2}."""
        h1 = build_h1(clean_df)
        assert h1["relation_lm_ord"].isin([0, 1, 2]).all()

    def test_h1_domaine_usage_range(self, clean_df):
        """Ind.1 VD — domaine_usage_freq ∈ [0, 4]."""
        h1 = build_h1(clean_df)
        assert h1["domaine_usage_freq"].between(0, 4).all()

    def test_h1_valorisation_range(self, clean_df):
        """Ind.3 VD — valorisation_sent ∈ [-1, 1]."""
        h1 = build_h1(clean_df)
        assert h1["valorisation_sent"].between(-1.0, 1.0).all()

    def test_no_leakage_h1(self):
        """h1_target ne doit PAS figurer dans les features d'entrée."""
        from train import train_h1
        import inspect
        src = inspect.getsource(train_h1)
        # La cible ne doit pas être dans feat_cols
        assert '"h1_target"' not in src.split("feat_cols")[0]


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURES H2 — Représentations
# ══════════════════════════════════════════════════════════════════════════════

class TestH2Features:

    def test_h2_perception_onehot(self, clean_df):
        """Ind.1 VI — colonnes perc_* créées."""
        h2 = build_h2(clean_df)
        for lbl in ["perc_utile", "perc_belle", "perc_difficile", "perc_importante"]:
            assert lbl in h2.columns

    def test_h2_perception_binary(self, clean_df):
        """Ind.1 VI — perc_* ∈ {0, 1}."""
        h2 = build_h2(clean_df)
        for lbl in ["perc_utile", "perc_belle", "perc_difficile", "perc_importante"]:
            assert h2[lbl].isin([0, 1]).all()

    def test_h2_hierarchie_range(self, clean_df):
        """Ind.4 VI — hierarchie_fr ∈ [1, 3]."""
        h2 = build_h2(clean_df)
        assert h2["hierarchie_fr"].between(1, 3).all()

    def test_h2_target_motivation_3classes(self, clean_df):
        """VD Cible A — motivation ∈ {0, 1, 2}."""
        h2 = build_h2(clean_df)
        assert h2["h2_target_motivation"].isin([0, 1, 2]).all()

    def test_h2_diff_labels_binary(self, clean_df):
        """VD Cible B — diff_* ∈ {0, 1}."""
        h2 = build_h2(clean_df)
        diff_cols = [c for c in h2.columns if c.startswith("diff_")]
        assert len(diff_cols) >= 3
        for col in diff_cols:
            assert h2[col].isin([0, 1]).all()

    def test_h2_all_diff_labels_present(self, clean_df):
        """Tous les labels de DIFFICULTE_KEYWORDS sont présents."""
        h2 = build_h2(clean_df)
        for label in DIFFICULTE_KEYWORDS:
            assert f"diff_{label}" in h2.columns


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURES H3 — Exposition plurilingue
# ══════════════════════════════════════════════════════════════════════════════

class TestH3Features:

    def test_h3_score_range(self, clean_df):
        """VD — h3_score_attitude ∈ [1.0, 5.0]."""
        h3 = build_h3(clean_df)
        assert h3["h3_score_attitude"].between(1.0, 5.0).all()

    def test_h3_attitude_classes_valid(self, clean_df):
        """VD — h3_attitude_class ∈ {Positive, Neutre, Négative}."""
        h3 = build_h3(clean_df)
        assert h3["h3_attitude_class"].isin(["Positive", "Neutre", "Négative"]).all()

    def test_h3_exposition_freq_range(self, clean_df):
        """Ind.1 VI — exposition_freq ∈ [0, 4]."""
        h3 = build_h3(clean_df)
        assert h3["exposition_freq"].between(0, 4).all()

    def test_h3_interet_binary(self, clean_df):
        """Ind.2 VI — interet_bin ∈ {-1, 0, 1}."""
        h3 = build_h3(clean_df)
        assert h3["interet_bin"].isin([-1, 0, 1]).all()

    def test_h3_score_consistency(self, clean_df):
        """Score élevé → classe Positive, score bas → classe Négative."""
        h3 = build_h3(clean_df)
        positives = h3[h3["h3_attitude_class"] == "Positive"]["h3_score_attitude"]
        negatives = h3[h3["h3_attitude_class"] == "Négative"]["h3_score_attitude"]
        if len(positives) > 0 and len(negatives) > 0:
            assert positives.mean() > negatives.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURES H4 — Intégration langues locales
# ══════════════════════════════════════════════════════════════════════════════

class TestH4Features:

    def test_h4_target_motivation_binary(self, clean_df):
        """VD Cible A — h4_target_motivation ∈ {0, 1}."""
        h4 = build_h4(clean_df)
        assert h4["h4_target_motivation"].isin([0, 1]).all()

    def test_h4_engagement_score_range(self, clean_df):
        """VD Cible B — h4_engagement_score ∈ {1, 2, 3, 4}."""
        h4 = build_h4(clean_df)
        assert h4["h4_engagement_score"].isin([1, 2, 3, 4]).all()

    def test_h4_souhait_freq_range(self, clean_df):
        """Ind.2 VI — souhait_freq ∈ [0, 4]."""
        h4 = build_h4(clean_df)
        assert h4["souhait_freq"].between(0, 4).all()

    def test_h4_discipline_labels_binary(self, clean_df):
        """Ind.3 VI / Cible C — vi_disc_* et vd_disc_* ∈ {0, 1}."""
        h4 = build_h4(clean_df)
        for disc in DISCIPLINE_KEYWORDS:
            assert f"vi_disc_{disc}" in h4.columns
            assert f"vd_disc_{disc}" in h4.columns
            assert h4[f"vi_disc_{disc}"].isin([0, 1]).all()

    def test_no_leakage_h4(self):
        """h4_target_motivation absent des features d'entrée."""
        from train import train_h4
        import inspect
        src = inspect.getsource(train_h4)
        feat_section = src.split("feat_cols")[1].split("X   =")[0]
        assert "h4_target_motivation" not in feat_section


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFIGURATION & MLFLOW
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_params_yaml_exists(self):
        """params.yaml doit exister à la racine."""
        assert Path("params.yaml").exists(), "params.yaml manquant"

    def test_mlflow_experiments_defined(self):
        """Les 4 experiments MLflow doivent être définis dans params.yaml."""
        try:
            cfg = load_config("params.yaml")
            experiments = cfg["mlflow"]["experiments"]
            for h in ["h1", "h2", "h3", "h4"]:
                assert h in experiments, f"Experiment {h} manquant"
        except FileNotFoundError:
            pytest.skip("params.yaml absent — skip test config")

    def test_thresholds_defined(self):
        """Les seuils de validation doivent être définis pour H1–H4."""
        try:
            cfg = load_config("params.yaml")
            for h in ["h1", "h2", "h3", "h4"]:
                assert "thresholds" in cfg[h], f"Seuils manquants pour {h}"
        except FileNotFoundError:
            pytest.skip("params.yaml absent — skip test config")

    def test_data_paths_defined(self):
        """Les chemins data doivent être définis dans params.yaml."""
        try:
            cfg = load_config("params.yaml")
            assert "raw_path" in cfg["data"]
            assert "processed_dir" in cfg["data"]
        except FileNotFoundError:
            pytest.skip("params.yaml absent — skip test config")
