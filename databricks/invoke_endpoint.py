# invoke_endpoint.py — Test the live FLP Model Serving endpoint from your machine
# Usage:  python databricks/invoke_endpoint.py all      (or h1, h2a, h2b, h3r, h3c, h4a, h4b, h4c)
# Requires: pip install requests
import json
import os
import sys
import unicodedata

import requests

HOST = "https://dbc-9e268203-7090.cloud.databricks.com"
ENDPOINT = f"{HOST}/serving-endpoints/flp-all-models/invocations"


def _load_token() -> str:
    """Token from DATABRICKS_TOKEN env var, else from ~/.databrickscfg [DEFAULT]."""
    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    if token:
        return token
    cfg = os.path.expanduser("~/.databrickscfg")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            section = None
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                elif section == "DEFAULT" and line.lower().startswith("token"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "No Databricks token found. Set the DATABRICKS_TOKEN environment "
        "variable or run 'databricks configure --token'."
    )

FEATURES = [
    "age", "apprent_anterieur_bin", "domaine_usage_freq", "exposition_freq",
    "hierarchie_fr", "importance_bin", "importance_sent", "interet_bin",
    "interet_camarades_bin", "interet_camarades_sent", "interet_sent",
    "lm_bassa", "lm_bassa'a", "lm_bulu", "lm_eton", "lm_ewondo", "lm_foufoulde",
    "lm_fufulde", "lm_ghomala", "lm_haoussa", "lm_mafa", "mots_assoc_sent",
    "nb_langues", "perc_belle", "perc_difficile", "perc_importante", "perc_utile",
    "perception_multi_ord", "perception_multi_sent", "region_Adamaoua",
    "region_Adamy", "region_Centrafrique", "region_Centre", "region_Centre_Afrique",
    "region_Centre_Est", "region_Cet", "region_Est", "region_Extreme_Nord",
    "region_Extreme-Nord", "region_Francais", "region_Gabon", "region_L'Est",
    "region_Littoral", "region_Niger", "region_Nigeria", "region_Nord",
    "region_Ouest", "region_Republique_Centrafricaine", "region_Sud",
    "region_Sud_Ouest", "region_Tchad", "relation_lm_ord", "sexe_bin",
    "souhait_freq", "valorisation_sent", "vi_disc_conjugaison",
    "vi_disc_expression_orale", "vi_disc_grammaire", "vi_disc_lecture",
    "vi_disc_vocabulaire",
]

LABELS = {
    "h1_pred": "H1 usage quotidien (0=NON, 1=OUI)",
    "h1_proba": "H1 probabilité OUI",
    "h2a_pred": "H2 motivation (0=faible, 1=moyen, 2=élevé)",
    "h2a_proba_max": "H2 motivation confiance",
    "h2b_0": "H2 difficulté: grammaire",
    "h2b_1": "H2 difficulté: vocabulaire",
    "h2b_2": "H2 difficulté: orthographe",
    "h2b_3": "H2 difficulté: conjugaison",
    "h2b_4": "H2 difficulté: expression orale",
    "h2b_5": "H2 difficulté: compréhension",
    "h2b_6": "H2 difficulté: analyse",
    "h3_reg": "H3 score attitude (1.0-5.0)",
    "h3_clf": "H3 attitude (0=négative, 1=neutre, 2=positive)",
    "h4a_pred": "H4 motivation (0/1)",
    "h4a_proba": "H4 probabilité motivation",
    "h4b_engagement": "H4 engagement (1-4)",
    "h4c_0": "H4 discipline: vocabulaire",
    "h4c_1": "H4 discipline: grammaire",
    "h4c_2": "H4 discipline: lecture",
    "h4c_3": "H4 discipline: expression orale",
    "h4c_4": "H4 discipline: conjugaison",
}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if model_name not in ("all", "h1", "h2a", "h2b", "h3r", "h3c", "h4a", "h4b", "h4c"):
        print(f"Unknown model '{model_name}'. Use: all, h1, h2a, h2b, h3r, h3c, h4a, h4b, h4c")
        sys.exit(1)

    record = {c: 0.0 for c in FEATURES}
    record["model_name"] = model_name
    record["age"] = 14.0
    record["nb_langues"] = 3.0
    record["lm_ewondo"] = 1.0
    record["region_Centre"] = 1.0
    record["domaine_usage_freq"] = 2.0
    record["exposition_freq"] = 3.0
    record["perc_utile"] = 1.0
    record["interet_camarades_bin"] = 1.0

    token = _load_token()
    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [record]},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    prediction = resp.json()["predictions"][0]
    print(f"=== Live prediction (model={model_name}) ===")
    for key, value in prediction.items():
        label = LABELS.get(key, key)
        if value != 0.0 or key in ("h1_pred", "h2a_pred", "h3_clf", "h3_reg", "h4a_pred", "h4b_engagement"):
            print(f"  {label:40s} {value}")


if __name__ == "__main__":
    main()
