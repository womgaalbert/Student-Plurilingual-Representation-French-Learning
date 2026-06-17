# Social Media Posts — FLP Dashboard Launch
# Style : Albert Womga — technique, chaleureux, structuré

## LinkedIn — Albert Womga (EN)

🌍 The Streamlit dashboard is LIVE — try it now
   — From PhD research to interactive ML app —

A few weeks ago I shared the first results of our collaboration with
Chancelline Armelle Nongni Kendjio. The pipeline is now deployed.

🎛️ Interactive dashboard → try any student profile :
👉 https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/

Chancelline collected data from 500 students across 15 schools in Cameroon.
Her question : "Does the plurilingual background of Cameroonian students
influence how they learn French?"

The dashboard lets you explore her findings — live.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 WHAT YOU CAN DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ Predict any student profile across H1, H2, H3, H4
  → adjust age, languages spoken, attitudes, motivation…
  → see the model's prediction + probability in real time

◆ Explore descriptive NLP analysis
  → CamemBERT embeddings + UMAP clusters
  → A Priori categories (Moscovici & Jodelet framework)
  → Stereotype detection, co-occurrence networks, n-grams

◆ Bilingual UI — toggle FR / EN at any time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 WHERE WE LANDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ H1 — Multilingual Repertoire → Daily Language Use
   XGBoost | F1=0.835 ✅ | AUC=0.851 ✅

◆ H2 — French Perceptions → Motivation & Difficulties
   ClassifierChain + XGBoost | F1=0.954 ✅ | F1-micro=0.745 ✅

◆ H3 — Plurilingual Exposure → Attitude toward French
   VotingRegressor (ET + XGBoost) | MAE=0.513 ⚠️ | F1=0.780 ✅
   CamemBERT PCA 20D integrated — MAE improved -3.4%

◆ H4 — Local Language Integration → Engagement
   XGBoost multi-task | F1=0.807 ✅ | Spearman ρ=0.561 ✅
   CamemBERT decisive here : ρ 0.476 → 0.561 (+0.085)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ MLOps — LEVEL 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FastAPI — 8 prediction endpoints on port 8001
✅ MLflow Model Registry — 7 models tracked
✅ Docker Compose — API + MLflow server
✅ GitHub Actions — CI/CD pipeline (42 tests, auto-deploy)
✅ Streamlit Cloud — public dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛠️ STACK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Python · Scikit-learn · XGBoost · LightGBM · CamemBERT (INRIA)
MLflow · FastAPI · Docker · Streamlit · GitHub Actions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3/4 hypotheses validated. H3 near the threshold (MAE 0.513 vs 0.50 target).
93.5% of students already exposed to other languages — near-zero variance
for causal testing. A finding in itself.

From a CSV file and a research question to a deployed ML app
supporting a doctoral thesis on the ground in Cameroon.

Honoured to build the technical side of Chancelline's journey 🙏

🔗 Dashboard : https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/
📂 GitHub : https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning
👩‍🎓 Chancelline Armelle Nongni Kendjio : https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/

Working on NLP, African languages or ML for social sciences?
Let's connect 🤝

#MachineLearning #NLP #CamemBERT #MLOps #XGBoost #MLflow
#DataScience #Sociolinguistics #Cameroon #Streamlit #FastAPI
#Education #Plurilingualism #AfricaTech #Research #PhD

---

## LinkedIn — Chancelline Armelle (FR)

🌍 Mon dashboard de recherche est en ligne — venez l'essayer !

Dans le cadre de ma thèse doctorale à l'Université Marie & Louis Pasteur
de Besançon, sous la direction de Serge Borg, j'ai collecté les réponses
de 500 élèves du secondaire dans 15 établissements au Cameroun.

Ma question de recherche :
« Le contexte plurilingue des élèves camerounais influence-t-il
leur apprentissage du français ? »

Aujourd'hui, les résultats sont accessibles à tous via un dashboard
interactif développé avec Albert Womga 👇
👉 https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 CE QUE VOUS POUVEZ FAIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ Simuler le profil d'un élève — âge, langues parlées, attitudes —
  et voir la prédiction du modèle en temps réel

◆ Explorer les analyses textuelles — nuages de mots,
  cartes UMAP, stéréotypes détectés, réseaux de co-occurrences

◆ Naviguer entre le français et l'anglais
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 RÉSULTATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ H1 — Le répertoire multilingue prédit l'usage quotidien ✅
◆ H2 — La perception du français prédit la motivation ✅
◆ H3 — L'exposition plurilingue influence l'attitude ⚠️
◆ H4 — L'intégration des langues locales favorise l'engagement ✅

3 hypothèses sur 4 validées — 7 modèles ML en production.
Un immense merci à Albert Womga pour avoir construit toute
l'infrastructure technique de ce projet 🙏

🔗 Dashboard : https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/
📂 Code source : https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning
👨‍💻 Albert Womga : https://www.linkedin.com/in/albert-womga-009a7931/

#Recherche #Linguistique #Cameroun #Plurilinguisme #These #FLES
#MachineLearning #NLP #Streamlit #DataScience #Education #Afrique

---

## X (Twitter) — Albert

🌍 LIVE: Interactive ML dashboard for a PhD on how Cameroonian students perceive learning French.

500 students · 15 schools · 4 hypotheses · 7 models

🔮 Try any student profile → instant prediction:
https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/

3/4 hypotheses validated ✅
CamemBERT + XGBoost + Streamlit

#MachineLearning #NLP #Cameroon #Streamlit #MLOps

---

## X (Twitter) — Chancelline

🌍 Mon dashboard de recherche est en ligne !

500 élèves · 15 établissements · 4 hypothèses
Le plurilinguisme influence-t-il l'apprentissage du français ?

🔮 Faites vos propres simulations :
https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/

#These #Cameroun #Plurilinguisme #IA #Recherche

---

## Instructions de publication

1. **Albert** poste la version LinkedIn EN en premier (portée internationale + technique)
2. **Chancelline** poste la version LinkedIn FR dans la foulée (public francophone, recherche)
3. **Images** : 2-3 captures du dashboard (page d'accueil, une prédiction H1, une carte UMAP)
4. **Alt-text** : Décrire chaque capture pour l'accessibilité
5. **X/Twitter** : Poster après LinkedIn, avec lien + capture
6. **Taguer** : @Streamlit, les universités, laboratoires
