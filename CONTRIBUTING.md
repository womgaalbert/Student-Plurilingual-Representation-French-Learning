# Contributing — French Learning Perceptions in Plurilingual Cameroon

Projet de recherche open source. Les contributions sont les bienvenues, en particulier autour du NLP français (fine-tuning CamemBERT), des jeux de données en langues camerounaises, de l'IA éducative et de la recherche sur le plurilinguisme.

---

## Stratégie de branches

- `main`      — code stable et testé uniquement
- `dev`       — branche d'intégration
- `feature/*` — nouvelles fonctionnalités
- `exp/*`     — expérimentations ML (ex: `exp/h1-lightgbm`)

---

## Avant de soumettre une PR

1. Exécuter `pytest tests/ -v` — tous les tests doivent passer
2. Les hyperparamètres doivent être dans `params.yaml`, pas en dur
3. Chaque run d'entraînement doit être loggé dans MLflow
4. Les données brutes ne doivent jamais être commitées (voir .gitignore)

---

## Analyse descriptive textuelle — Résultats

### Profil démographique (N = 500)

![Profil démographique](reports/demographics/demographics_overview.png)

*Figure 1 — Profil démographique global : âge (moy. 12,9 ans), classe (majorité 6e/5e), région (Centre dominant), genre (~300 F / ~200 M), établissement (LGL, Ste Famille, Lycée Leclerc en tête) et langue maternelle (Ewondo ~85, Eton ~27, Fufuldé, Bulu).*

![Langues parlées](reports/demographics/demographics_langues.png)

*Figure 2 — Top-20 des langues parlées (réponses multiples) : le français domine (~305), suivi de l'anglais (~97), puis les langues nationales camerounaises (Ewondo, Foufouldé, Fufuldé, Haoussa). La quasi-totalité des élèves parle ≥2 langues — le plurilinguisme est la norme.*

---

### N-grammes — H1 (Répertoire multilingue & mobilisation)

Question source : **« Pensez-vous que parler plusieurs langues est un avantage ? Pourquoi ? »**

**Top-20 unigrammes** :

![Unigrammes H1](reports/ngrams_h1_unigrammes.png)

*Figure 3 — Top-20 unigrammes H1 : « oui » (475 occurrences) domine massivement, suivi de « complexe » (200), « aisé » (121), « parcequ » (69), « langue » (62). Les termes « communiquer » (59), « voyager » (43) et « aider » (28) révèlent les motivations principales : communication, mobilité, entraide. Les noms de langues nationales (foufouldé, ewondo) ancrent le discours dans des pratiques locales concrètes.*

**Top-20 bigrammes** :

![Bigrammes H1](reports/ngrams_h1_bigrammes.png)

*Figure 4 — Top-20 bigrammes H1 : la structure discursive dominante est « OUI + justification » (« oui parcequ »=62, « oui parceque »=35, « oui aisé »=34, « oui voyager »=27, « oui communiquer »=22). La polarisation du discours autour de « oui » et dans une moindre mesure « non + personne » est clairement visible.*

**Top-20 trigrammes** :

![Trigrammes H1](reports/ngrams_h1_trigrammes.png)

*Figure 5 — Top-20 trigrammes H1 : « oui parcequ causer », « oui pouvoir voyager » et « oui langue étranger » forment les constructions argumentatives les plus fréquentes — la communication interpersonnelle et l'ouverture au monde sont les arguments centraux.*

---

### N-grammes — H2 (Représentations du français)

Questions sources fusionnées : **« Aspects difficiles du français » + « Principales difficultés » + « Origine des difficultés »**

**Top-20 unigrammes** :

![Unigrammes H2](reports/ngrams_h2_unigrammes.png)

*Figure 6 — Top-20 unigrammes H2 : « vocabulaire » (493), « compréhension » (489), « texte » (489), « grammaire » (488), « orthographe » (487), « lecture » (486) et « expression orale » (484) forment un bloc quasi homogène en tête. Contrairement à l'hypothèse initiale qui attendait grammaire et conjugaison seules en tête, les difficultés sont réparties sur les QUATRE compétences langagières : lexique, compréhension écrite, grammaire et oral. Ce constat a conduit à l'ajout du mot-clé « compréhension » dans `DIFFICULTE_KEYWORDS`.*

**Top-20 bigrammes** :

![Bigrammes H2](reports/ngrams_h2_bigrammes.png)

*Figure 7 — Top-20 bigrammes H2 : les collocations « grammaire + orthographe » et « expression + oral » confirment que ces difficultés sont souvent mentionnées ensemble par les élèves. La co-occurrence fréquente justifie l'utilisation du ClassifierChain pour capturer ces dépendances.*

---

### N-grammes — H3 (Exposition plurilingue & attitudes)

**Top-20 unigrammes** :

![Unigrammes H3](reports/ngrams_h3_unigrammes.png)

*Figure 8 — Top-20 unigrammes H3 : « bien » domine massivement, confirmant l'approbation quasi-générale du plurilinguisme. L'association « langue + culture + français » suggère que les élèves articulent leur rapport au français dans un cadre culturel élargi, pas purement scolaire. Les termes « voyager », « expliquer », « étranger », « histoire » témoignent d'une vision positive et ouverte.*

---

### N-grammes — H4 (Intégration des langues locales)

**Top-20 unigrammes** :

![Unigrammes H4](reports/ngrams_h4_unigrammes.png)

*Figure 9 — Top-20 unigrammes H4 : « grammaire » et « orthographe » dominent, suivis de « conjugaison », « vocabulaire » et « oral ». Grammaire et orthographe sont les obstacles perçus n°1 et n°2, mais conjugaison et vocabulaire sont les disciplines où les élèves souhaitent prioritairement l'intégration des langues camerounaises.*

**Top-20 bigrammes et trigrammes** :

![Bigrammes H4](reports/ngrams_h4_bigrammes.png)

*Figure 10 — Bigrammes H4 : les collocations confirment les associations entre disciplines — « grammaire + orthographe », « conjugaison + vocabulaire ».*

![Trigrammes H4](reports/ngrams_h4_trigrammes.png)

*Figure 11 — Trigrammes H4 : les formules à 3 mots révèlent des combinaisons de difficultés et de souhaits d'intégration.*

---

## CamemBERT — Modèle, usage et résultats

### Spécifications techniques

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `camembert-base` (110M paramètres, 768 dimensions d'embedding) |
| Type | Modèle de langue français pré-entraîné par couche transformer (12 couches, 12 têtes d'attention) |
| Entraînement original | Sous-corpus français d'OSCAR (138 Go de texte) |
| Tâche dans FLP | Embeddings contextuels → réduction UMAP → clustering K-Means |
| Utilisation | Uniquement en inference (pas de fine-tuning pour l'instant) |
| Batch size | 16 textes (limité par la RAM CPU) |
| Max length | 128 tokens par texte |
| Device | CPU (pas de GPU disponible dans l'environnement actuel) |
| Temps d'encodage | ~2 secondes pour 500 textes sur CPU |

### Pipeline NLP complet

```
Texte brut
  └─→ spaCy (fr_core_news_sm) → lemmatisation + suppression stop words
       └─→ CamemBERT (camembert-base) → vecteur 768 dimensions
            └─→ UMAP (n_neighbors=15, min_dist=0.1) → projection 2D
                 └─→ K-Means (k=5) → 5 clusters thématiques
```

---

### Wordclouds — Nuages de mots par hypothèse

#### H1 — Nuage de mots global

![Wordcloud H1 Global](reports/h1/descriptive/wordcloud_global.png)

*Figure 12 — Nuage de mots H1 : « oui », « langue », « communiquer », « parcequ » et « foufoulde » dominent. Les élèves justifient majoritairement leur plurilinguisme par des raisons communicatives (« communiquer avec les étrangers », « voyager », « aider les autres »).*

#### H1 — Wordclouds par catégorie A Priori

| Représentation | Nuage de mots |
|---|---|
| **Utilitaire** | ![WC Utilitaire](reports/h1/descriptive/wordcloud_representation_utilitaire.png) |
| **Identitaire** | ![WC Identitaire](reports/h1/descriptive/wordcloud_representation_identitaire.png) |
| **Affective** | ![WC Affective](reports/h1/descriptive/wordcloud_representation_affective.png) |
| **Institutionnelle** | ![WC Institutionnelle](reports/h1/descriptive/wordcloud_representation_institutionnelle.png) |
| **Résistance / Contrainte** | ![WC Resistance](reports/h1/descriptive/wordcloud_resistance_contrainte.png) |

*Figures 13-17 — Nuages de mots H1 par catégorie A Priori. Chaque nuage révèle le vocabulaire caractéristique de la catégorie : l'identitaire mobilise les noms de langues et les marqueurs d'appartenance ; l'utilitaire mobilise la réussite, le travail, l'avenir ; la contrainte mobilise l'obligation, la force, l'imposition.*

#### H2 — Nuage de mots global

![Wordcloud H2 Global](reports/h2/descriptive/wordcloud_global.png)

*Figure 18 — Nuage de mots H2 : « grammaire » et « orthographe » dominent en grande taille, confirmant leur statut de difficultés centrales. « Vocabulaire », « expression », « compréhension » et « lecture » apparaissent en tailles intermédiaires. « Conjugaison » est bien présente mais moins massive que prévu.*

#### H3 — Nuage de mots global

![Wordcloud H3 Global](reports/h3/descriptive/wordcloud_global.png)

*Figure 19 — Nuage de mots H3 : « bien », « langue », « culture », « national » et « français » dominent. La co-présence de « français » et « culture » indique que les élèves articulent leur rapport au français dans un cadre culturel élargi — cohérent avec des attitudes globalement positives.*

#### H4 — Nuage de mots global

![Wordcloud H4 Global](reports/h4/descriptive/wordcloud_global.png)

*Figure 20 — Nuage de mots H4 : « grammaire » et « orthographe » dominent massivement, avec « vocabulaire », « expression », « bien » et « conjugaison » bien visibles. La dominance de la grammaire confirme que les élèves souhaitent prioritairement l'intégration des langues locales dans cette discipline.*

---

### Analyse spatiale — Projections UMAP

#### UMAP par hypothèse — Clusters K-Means (k=5)

| H1 — Clusters | H2 — Clusters |
|---|---|
| ![UMAP H1 Clusters](reports/h1/descriptive/umap_clusters.png) | ![UMAP H2 Clusters](reports/h2/descriptive/umap_clusters.png) |

*Figures 21-22 — **H1** : la grande majorité des répondants se concentre dans une zone dense (clusters 1-4 très proches), tandis que le cluster 0 contient quelques points très isolés. Cette séparation nette suggère un discours homogène sur le plurilinguisme. **H2** : 5 clusters bien séparés et relativement équilibrés, indiquant une plus grande diversité des profils de difficulté (grammaire/conjugaison vs vocabulaire/compréhension vs sans difficulté majeure).*

| H3 — Clusters | H4 — Clusters |
|---|---|
| ![UMAP H3 Clusters](reports/h3/descriptive/umap_clusters.png) | ![UMAP H4 Clusters](reports/h4/descriptive/umap_clusters.png) |

*Figures 23-24 — **H3** : structure intermédiaire entre H1 (très concentrée) et H2 (très dispersée) — cohérent avec l'exposition quasi-universelle (93,5% exposés). **H4** : structure la plus dispersée de toutes les hypothèses, reflétant une forte hétérogénéité des profils d'engagement et de disciplines souhaitées.*

#### UMAP — Stéréotypes détectés

| H1 — Stéréotypes | H2 — Stéréotypes |
|---|---|
| ![UMAP H1 Stereotypes](reports/h1/descriptive/umap_stereotypes.png) | ![UMAP H2 Stereotypes](reports/h2/descriptive/umap_stereotypes.png) |

*Figures 25-26 — Projections UMAP colorées par stéréotype détecté. En H1, très peu de stéréotypes sont détectés (discours sain). En H2, les préconstruits de difficulté apparaissent de manière plus marquée, colorant une partie significative de l'espace.*

#### UMAP — Catégories A Priori

| H1 — A Priori | H2 — A Priori |
|---|---|
| ![UMAP H1 A Priori](reports/h1/descriptive/umap_apriori.png) | ![UMAP H2 A Priori](reports/h2/descriptive/umap_apriori.png) |

*Figures 27-28 — Projections UMAP colorées par catégorie A Priori dominante. En H1, la représentation identitaire (orange) domine largement l'espace. En H2, la résistance/contrainte (rouge) occupe une zone plus étendue, cohérent avec le fait que le français est davantage perçu comme une imposition institutionnelle.*

---

### Analyse A Priori — Distribution (Cadre Moscovici & Jodelet)

Classification automatique des réponses en 5 catégories de représentations sociales via **similarité cosinus** entre les embeddings CamemBERT des réponses élèves et les phrases de référence (protoréponses) définies pour chaque catégorie.

| Catégorie A Priori | Protypes de référence | H1 (distribution) | H2 (distribution) |
|---|---|---|---|
| Représentation **utilitaire** | *nécessité, réussir, travail, utile, avenir* | Minoritaire (~30) | Modérée |
| Représentation **identitaire** | *langue des autres, pas la mienne, étrangers, colons* | **Dominante (~270)** | **Dominante** |
| Représentation **affective** | *j'aime, j'ai peur, difficile, passion, belle* | Très minoritaire (~12) | Minoritaire |
| Représentation **institutionnelle** | *école, professeur, obligatoire, cours, règle* | Faible | Modérée |
| Résistance / **contrainte** | *forcé, obligé, pas le choix, contraint, imposé* | Forte (~190) | Proportionnellement plus forte que H1 |

**Distribution visuelle :**

| H1 — Distribution A Priori | H2 — Distribution A Priori |
|---|---|
| ![A Priori H1](reports/h1/descriptive/apriori_distribution.png) | ![A Priori H2](reports/h2/descriptive/apriori_distribution.png) |

*Figures 29-30 — Distribution des catégories A Priori. **H1** : la représentation identitaire domine (~270), suivie de la résistance/contrainte (~190), l'utilitaire (~30) et l'affective (~12) étant minoritaires. **H2** : la résistance/contrainte est proportionnellement plus forte, reflétant l'ambivalence des élèves envers le français — langue imposée par l'institution scolaire, mais reconnue comme importante pour l'avenir.*

> **Interprétation clé** : La représentation **identitaire** domine dans les deux hypothèses. Les élèves perçoivent le plurilinguisme (H1) et le français (H2) principalement à travers le prisme de l'identité — « c'est ma langue / ce n'est pas la mienne ». La forte présence de la résistance/contrainte en H2 signale que le français est vécu par une partie significative des élèves comme une imposition institutionnelle.

---

### Analyse des stéréotypes et préconçus

6 marqueurs de stéréotypes sont détectés automatiquement par double méthode : **keyword matching** (expressions régulières) ET **cosine similarity** entre les embeddings CamemBERT des réponses et des phrases stéréotypiques de référence.

| Marqueur | Phrases de référence | Méthode |
|---|---|---|
| Distanciation identitaire | *langue des colons, langue des blancs, langue étrangère* | Keywords + Embedding |
| Préconstruit de difficulté | *trop difficile, trop dur, très difficile* | Keywords + Embedding |
| Préconstruit d'inutilité | *inutile dans ma vie, pas utile, sert à rien* | Keywords + Embedding |
| Auto-dévalorisation | *nul en français, mauvais en français, pas bon* | Keywords + Embedding |
| Exclusion symbolique | *pas pour nous, pas pour moi, pas notre langue* | Keywords + Embedding |
| Contrainte / résistance | *obligés d'apprendre, pas le choix, forcés à* | Keywords + Embedding |

| Stéréotype | H1 | H2 | Interprétation |
|---|---|---|---|
| Exclusion symbolique | 2 élèves | Faible | « Pas pour nous / pas notre langue » — très rare, discours globalement inclusif |
| Contrainte / résistance | Modéré | Élevé | Cohérent avec la dominance de la catégorie A Priori « résistance_contrainte » |
| Préconstruit de difficulté | Faible | Très élevé | Spécifique à H2 : « trop difficile / trop dur » — 200 occurrences de « complexe » en unigrammes |
| Auto-dévalorisation | Très faible | Faible | « Je suis nul(le) en français » — rare mais préoccupant sur le plan psychopédagogique |

**Distribution visuelle :**

| H1 — Stéréotypes | H2 — Stéréotypes |
|---|---|
| ![Stereo H1](reports/h1/descriptive/stereotype_distribution.png) | ![Stereo H2](reports/h2/descriptive/stereotype_distribution.png) |

*Figures 31-32 — Distribution des stéréotypes détectés. **H1** : seuls 2 élèves présentent un stéréotype d'exclusion symbolique — l'échantillon présente un discours globalement sain. **H2** : la contrainte/résistance et le préconstruit de difficulté sont les marqueurs dominants, cohérents avec les résultats de l'analyse A Priori.*

> **Interprétation** : L'échantillon présente un discours globalement sain, sans préjugé linguistique massif. L'auto-dévalorisation, bien que rare, est le marqueur le plus préoccupant sur le plan psychopédagogique.

---

### Réseaux de co-occurrences — H1

![Heatmap H1](reports/h1/descriptive/heatmap_cooccurrence.png)

*Figure 33 — Heatmap de co-occurrence H1 : la cellule (oui, oui) est la plus foncée (~230 co-occurrences), suivie de (non, non). La ligne « oui » présente des co-occurrences modérées avec « communiquer », « foufoulde », « langue », « parcequ » — confirmant le lien entre plurilinguisme actif et motivation communicative.*

![Réseau H1](reports/h1/descriptive/network_cooccurrence.png)

*Figure 34 — Réseau de co-occurrences H1 : graphe centré sur « oui » et « non » comme hubs. « Oui » rayonne vers « communiquer », « voyager », « aider », « pays » et les noms de langues nationales. « Non » est connecté à « personne » et « pouvoir ». Cette structure en étoile illustre la polarisation du discours autour de deux positions distinctes.*

---

## État du pipeline ML

### Performances par hypothèse (Mai 2026)

| Hypothèse | Tâche ML | Métrique clé | Seuil | Valeur | Statut |
|-----------|----------|-------------|-------|--------|--------|
| H1 | Classification binaire | F1-macro / ROC-AUC | ≥0.70 / ≥0.75 | 0.835 / 0.851 | ✅ Validée |
| H2 | Double cible (3-cl + multi-label) | F1-weighted / F1-micro | ≥0.65 / ≥0.72 | 0.954 / 0.745 | ✅ Validée |
| H3 | Régression + Causal | MAE / Pearson p | ≤0.50 / <0.05 | 0.531 / 0.984 | ⚠️ En cours |
| H4 | Triple cible (bin + ord + ml) | F1 / Spearman ρ / SubAcc | ≥0.70 / ≥0.55 / ≥0.45 | 0.80 / 0.476 / 1.0 | ⚠️ En cours |

### Maturité MLOps

| Niveau | Contenu | Statut |
|--------|---------|--------|
| Level 0 | Scripts manuels Python | ✅ Terminé |
| Level 1 | Pipeline automatisé + MLflow tracking | ✅ Terminé |
| Level 2 | CI/CD + Model Registry + FastAPI | 🔲 À faire |
| Level 3 | Monitoring drift (Evidently AI) + Auto-retraining | 🔲 À faire |

---

## Axes de contribution prioritaire

### 🔴 Haute priorité
- **CamemBERT features pour H3 & H4** : intégration des embeddings 768-dim (réduits par PCA) comme features d'entraînement
- **Amélioration du score d'engagement H4** : raffinement de la formule composite ou features comportementales supplémentaires
- **Variable d'exposition H3** : remplacer le binaire exposition_bin par une mesure plus granulaire

### 🟡 Priorité moyenne
- **Fine-tuning CamemBERT** : fine-tuner sur les réponses d'élèves camerounais pour de meilleurs embeddings
- **Modèles alternatifs H3** : tester LightGBM et ExtraTreesRegressor pour la régression
- **SHAP pour H1 et H2** : interprétabilité des features les plus contributives

### 🟢 Nice to have
- **LLM pour analyse qualitative** : zero-shot / few-shot classification des réponses libres
- **RAG pédagogique** : base documentaire FLES + retrieval pour recommandations contextualisées
- **Agent LLM** : génération automatique de rapports pédagogiques personnalisés par établissement
- **Jeux de données en langues camerounaises** : corpus parallèles pour les 250+ langues nationales
