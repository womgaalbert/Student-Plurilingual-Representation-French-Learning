"""
constants.py — French-Learning-Perceptions ML
Mappings colonnes CSV, encodages ordinaux, keywords ML.
"""

# Noms courts des colonnes (alignés table French-Learning-Perceptions)
COL_CONSENTEMENT      = "consentement"
COL_AGE               = "age"
COL_SEXE              = "sexe"
COL_REGION            = "region"
COL_ETABLISSEMENT     = "etablissement"
COL_CLASSE            = "classe"

# H1 — VI
COL_LANGUES_PARLEES   = "langues_parlees"
COL_LANGUE_MATERNELLE = "langue_maternelle"
COL_USAGE_QUOTIDIEN   = "usage_quotidien"        # → CIBLE H1
COL_APPRENTISSAGE_ANT = "apprentissage_anterieur"
COL_RELATION_LM       = "relation_langue_mat"
COL_AVANTAGE_MULTI    = "avantage_plurilingue"

# H2 — VI
COL_PERCEPTION_FR     = "perception_francais"
COL_MOTS_ASSOCIES     = "mots_associes"
COL_MOTIVATION        = "motivation_apprendre"
COL_FACILE            = "aspects_faciles"
COL_DIFFICILE         = "aspects_difficiles"
COL_IMPORTANCE_FR     = "importance_francais"
COL_COMPARAISON_FR    = "comparaison_francais"

# H3 — VI
COL_EXPOSITION        = "exposition_autres_langues"
COL_INTERET_APPRENDRE = "interet_autres_langues"
COL_PERCEPTION_MULTI  = "perception_plurilinguisme"
COL_MOTIVATION_CAMERO = "motivation_camerounaises"

# H4 — VI
COL_PARLE_COURS       = "langues_parlees_cours"
COL_INTERET_CAMARADES = "interet_langues_camarades"
COL_DIFFICULTES       = "difficultes_principales"
COL_ORIGINE_DIFF      = "origine_difficultes"
COL_INCLURE_LANGUES   = "souhait_inclure_langues"
COL_DISCIPLINE_ASSOC  = "discipline_associee"

# Mapping CSV original → noms courts
CSV_COLUMN_MAP = {
    "Demande de consentement pour participer à l'enquête": COL_CONSENTEMENT,
    "Informations personnelles \n\nÂge ?": COL_AGE,
    "sexe ?": COL_SEXE,
    "Région d'origine ?": COL_REGION,
    "Etablissement ?": COL_ETABLISSEMENT,
    "Classe ?": COL_CLASSE,
    "Quelles sont les langues que vous parlez?": COL_LANGUES_PARLEES,
    "Quelle est votre langue maternelle ?": COL_LANGUE_MATERNELLE,
    "Utilisez-vous d'autres langues dans votre quotidien ?": COL_USAGE_QUOTIDIEN,
    "Avez-vous appris d'autres langues avant le français ? Si oui, lesquelles ?": COL_APPRENTISSAGE_ANT,
    "Comment décririez-vous votre relation avec votre langue maternelle ?": COL_RELATION_LM,
    "Pensez-vous que parler plusieurs langues est un avantage ? Pourquoi ? (justifiez votre réponse)": COL_AVANTAGE_MULTI,
    "Comment percevez-vous la langue française ?": COL_PERCEPTION_FR,
    "citez quatre mots qui vous viennent à l'esprit quand vous pensez  au français:": COL_MOTS_ASSOCIES,
    "Qu'est-ce qui vous motive à apprendre le français ?_": COL_MOTIVATION,
    "Quels aspects du français trouvez - vous faciles ou difficiles à apprendre ? [Facile]": COL_FACILE,
    "Quels aspects du français trouvez - vous faciles ou difficiles à apprendre ? [Difficile]": COL_DIFFICILE,
    "Pensez-vous que le français est une langue importante dans le monde ? Pourquoi": COL_IMPORTANCE_FR,
    "Comment le français se compare-t-il aux autres langues que vous connaissez ou que vous entendez dans votre environnement ?": COL_COMPARAISON_FR,
    "Pendant vos cours ou dans votre vie quotidienne, avez-vous l'occasion d'entendre ou de parler d'autres langues que le français ?": COL_EXPOSITION,
    "Trouvez-vous intéressant ou utile d'en apprendre davantage sur d'autres langues en plus du français ? Pourquoi": COL_INTERET_APPRENDRE,
    "Comment percevez-vous le fait de parler plusieurs langues ?": COL_PERCEPTION_MULTI,
    "Pensez-vous que l'apprentissage du français en même temps que les langues camerounaises vous motiverait ?": COL_MOTIVATION_CAMERO,
    "Pendant vos cours, est-ce que l'on parle parfois des différentes langues des élèves ? Si oui, comment cela est-il fait ?": COL_PARLE_COURS,
    "Trouveriez-vous intéressant d'apprendre des choses sur les langues parlées par vos camarades de classe ?": COL_INTERET_CAMARADES,
    "Quelles sont les principales difficultés que vous rencontrez en apprenant le français ?": COL_DIFFICULTES,
    "Est-ce que ces difficultés sont liées à la langue elle-même, au vocabulaire, à la grammaire ou à d'autres aspects ?": COL_ORIGINE_DIFF,
    "Aimeriez-vous que les cours de français incluent davantage d'activités sur d'autres langues ?": COL_INCLURE_LANGUES,
    "A quelle discipline du français aimeriez-vous que l'apprentissage des langues camerounaises soit associé ?": COL_DISCIPLINE_ASSOC,
}

# Encodages ordinaux
FREQ_MAP = {
    "Toujours": 4, "toujours": 4,
    "Souvent": 3,  "souvent": 3,
    "Parfois": 2,  "parfois": 2,
    "Rarement": 1, "rarement": 1,
    "Jamais": 0,   "jamais": 0,
}
LIKERT_MAP = {
    "Très bien": 4,       "très bien": 4,
    "Plutôt bien": 3,     "plutôt bien": 3,
    "Un peu bien": 2,     "un peu bien": 2,
    "Pas du tout bien": 1,"pas du tout bien": 1,
}
IMPORTANCE_MAP = {
    "plus important": 3,  "Plus important": 3,
    "autant important": 2,"Autant important": 2,
    "moins important": 1, "Moins important": 1,
}
RELATION_LM_MAP = {"Aisé": 2, "aisé": 2, "Complexe": 1, "complexe": 1}

# Labels de perception du français (H2 — multi-label binaire)
PERCEPTION_LABELS = [
    "belle", "difficile", "importante", "utile", "compliquée",
    "intéressante", "ennuyeuse", "nécessaire", "riche", "obligatoire",
]

# Keywords multi-label H2 (difficultés)
DIFFICULTE_KEYWORDS = {
    "grammaire":        ["grammaire", "grammatical", "accord"],
    "vocabulaire":      ["vocabulaire", "mots", "lexique"],
    "orthographe":      ["orthographe", "fautes"],
    "conjugaison":      ["conjugaison", "conjuguer", "verbes"],
    "expression_orale": ["expression orale", "oral", "parler"],
    "analyse":          ["analyse", "analyser"],
}

# Keywords multi-label H4 (disciplines)
DISCIPLINE_KEYWORDS = {
    "vocabulaire":      ["vocabulaire", "mots", "lexique"],
    "grammaire":        ["grammaire"],
    "lecture":          ["lecture", "lire", "textes", "récit"],
    "expression_orale": ["expression orale", "oral", "parler"],
    "conjugaison":      ["conjugaison", "conjuguer"],
}

MOTIVATION_HIGH_KW = [
    "communication", "communiquer", "réussir", "avenir",
    "s'exprimer", "aider", "voyager", "études", "comprendre",
]
MOTIVATION_LOW_KW = ["obligé", "forcé", "difficile", "rien"]
