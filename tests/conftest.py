"""
conftest.py — Fixtures partagées French-Learning-Perceptions ML
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ajouter src/ au path Python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_df():
    """
    DataFrame minimal (10 répondants fictifs) pour tester le pipeline.
    Couvre toutes les colonnes requises H1–H4.
    """
    return pd.DataFrame({
        "consentement":              ["J'accepte"] * 9 + ["Je refuse"],
        "horodateur":                ["2025-05-14"] * 10,
        "age":                       ["11ans","12","13 ans","10","14","12","11","13","12","15"],
        "sexe":                      ["M","F","M","M","F","M","F","M","M","F"],
        "region":                    ["Ouest","Centre","Ouest","Centre","Sud",
                                      "Centre","Ouest","Centre","Ouest","Centre"],
        "etablissement":             ["La Retraite"] * 10,
        "classe":                    ["6e"] * 10,
        "langues_parlees":           ["Français, Anglais, Bamoun","Français",
                                      "Français, Douala","Français, Anglais",
                                      "Français, Ewondo","Français, Bulu",
                                      "Français, Anglais","Français, Dschang",
                                      "Français","Français"],
        "langue_maternelle":         ["Bamoun","Bafoussam","Douala","Bafang","Ewondo",
                                      "Bulu","Mangissa","Dschang","Bamoun","Ewondo"],
        "usage_quotidien":           ["OUI","OUI","NON","OUI","NON",
                                      "OUI","OUI","NON","OUI","NON"],
        "apprentissage_anterieur":   ["NON","NaN","NON","OUI","NON",
                                      "OUI","NON","NON","NaN","NON"],
        "relation_langue_mat":       ["Aisé","Complexe","Aisé","Aisé","Complexe",
                                      "Aisé","Complexe","Aisé","Aisé","Complexe"],
        "avantage_plurilingue":      ["Oui ça permet de communiquer",
                                      "Oui ça élargit les connaissances",
                                      "Oui pour voyager","Oui pour réussir",
                                      "Oui c'est important","Oui meilleure communication",
                                      "Oui c'est utile","Oui pour s'exprimer",
                                      "Oui ça aide","Oui pour s'intégrer"],
        "perception_francais":       ["utile","importante","belle","utile","difficile",
                                      "utile","importante","belle","utile","importante"],
        "mots_associes":             ["grammaire, lecture","conjugaison, vocabulaire",
                                      "analyse, orthographe","grammaire, vocabulaire",
                                      "difficile, fautes","utile, communication",
                                      "beau, important","difficile, grammaire",
                                      "vocabulaire, lecture","grammaire, conjugaison"],
        "motivation_apprendre":      ["Ça permet de communiquer","Pour réussir",
                                      "Belle langue","Pour voyager et travailler",
                                      "Mes parents parlent français","Pour m'exprimer",
                                      "Pour l'avenir","C'est éducatif",
                                      "Pour les études","Pour m'intégrer"],
        "aspects_faciles":           ["Lecture","Grammaire","Vocabulaire","Lecture",
                                      "Rien","Vocabulaire","Lecture","Grammaire",
                                      "Vocabulaire","Lecture"],
        "aspects_difficiles":        [None]*10,
        "importance_francais":       ["Oui c'est important","Oui","Oui","Oui","Oui",
                                      "Oui","Oui","Oui","Oui","Oui"],
        "comparaison_francais":      ["plus important","autant important","plus important",
                                      "moins important","autant important","plus important",
                                      "autant important","plus important","plus important",
                                      "autant important"],
        "exposition_autres_langues": ["Souvent","Parfois","Toujours","Rarement",
                                      "Souvent","Toujours","Parfois","Souvent",
                                      "Rarement","Toujours"],
        "interet_autres_langues":    ["Oui c'est intéressant","Oui ça élargit",
                                      "Oui parce que","Non pas vraiment",
                                      "Oui pour communiquer","Oui c'est utile",
                                      "Oui bien sûr","Oui ça aide",
                                      "Oui pour apprendre","Oui"],
        "perception_plurilinguisme": ["Très bien","Plutôt bien","Très bien",
                                      "Un peu bien","Plutôt bien","Très bien",
                                      "Plutôt bien","Très bien","Un peu bien","Très bien"],
        "motivation_camerounaises":  ["OUI","OUI","OUI","NON","OUI",
                                      "OUI","NON","OUI","OUI","OUI"],
        "interet_langues_camarades": ["Oui c'est bien","Oui","Un peu","Oui",
                                      "Oui bien sûr","Oui","Oui","Oui","Oui","Oui"],
        "difficultes_principales":   ["grammaire, conjugaison","vocabulaire","orthographe",
                                      "grammaire","conjugaison","vocabulaire","grammaire",
                                      "conjugaison","vocabulaire","orthographe"],
        "origine_difficultes":       ["la grammaire","le vocabulaire","l'orthographe",
                                      "la grammaire","la conjugaison","le vocabulaire",
                                      "la grammaire","la conjugaison","le vocabulaire",
                                      "l'orthographe"],
        "souhait_inclure_langues":   ["Toujours","Souvent","Parfois","Toujours",
                                      "Souvent","Toujours","Parfois","Souvent",
                                      "Toujours","Souvent"],
        "discipline_associee":       ["vocabulaire","grammaire","lecture",
                                      "vocabulaire, grammaire","expression orale",
                                      "grammaire","vocabulaire","lecture",
                                      "conjugaison","grammaire"],
        "langues_parlees_cours":     ["Oui l'anglais","Non","Oui","Non",
                                      "Oui l'anglais","Oui","Non","Oui","Non","Oui"],
    })


@pytest.fixture
def clean_df(sample_df):
    """DataFrame après filtre consentement + anonymisation + démographie."""
    from preprocess import filter_consent, anonymize, clean_demographics
    df = filter_consent(sample_df, "accepte")
    df = anonymize(df)
    df = clean_demographics(df)
    return df
