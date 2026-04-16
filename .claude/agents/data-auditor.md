# Agent : Data Auditor — French-Learning-Perceptions ML
# Exécuté automatiquement avant chaque pipeline (hook pre-train)

## Rôle
Vérification de la qualité des 500 répondants AVANT tout entraînement.
Accès lecture seule sur data/processed/.

## Checks obligatoires

### 1. Consentement
```python
assert df["consentement"].str.lower().str.contains("accepte").all()
```

### 2. Couverture colonnes par hypothèse
```python
REQUIRED = {
    "H1": ["nb_langues","apprent_anterieur_bin","relation_lm_ord",
            "domaine_usage_freq","valorisation_sent","h1_target"],
    "H2": ["perc_utile","perc_belle","mots_assoc_sent","hierarchie_fr",
            "h2_target_motivation","diff_grammaire"],
    "H3": ["exposition_freq","interet_bin","perception_multi_ord",
            "h3_score_attitude","h3_attitude_class"],
    "H4": ["interet_camarades_bin","souhait_freq","h4_target_motivation",
            "h4_engagement_score"],
}
for h, cols in REQUIRED.items():
    nan_rate = df[cols].isnull().mean()
    assert (nan_rate < 0.30).all(), f"Trop de NaN pour {h}"
```

### 3. Distribution régionale
```python
region_dist = df["region"].value_counts(normalize=True)
assert region_dist.max() < 0.60, "Déséquilibre régional > 60%"
```

### 4. Balance des cibles
```python
print("H1:", df["h1_target"].value_counts(normalize=True).to_dict())
print("H4:", df["h4_target_motivation"].value_counts(normalize=True).to_dict())
```

### 5. Rapport de sortie
```json
{
  "n_total": 500,
  "n_consentement_ok": 498,
  "nan_rates_par_hypothese": {"H1": 0.02, "H2": 0.05, "H3": 0.03, "H4": 0.04},
  "region_distribution": {"Ouest": 0.35, "Centre": 0.40},
  "h1_balance": {"OUI": 0.68, "NON": 0.32},
  "ready_for_training": true
}
```
