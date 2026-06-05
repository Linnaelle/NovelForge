# NovelForge - Genre Prediction from Synopsis

NovelForge est un projet de Deep Learning / NLP dont l'objectif est de predire les genres d'un Light Novel ou d'un Manhwa a partir de son synopsis.

Le projet couvre actuellement :

- **Jalon 2 - EDA et preprocessing** : analyse exploratoire, nettoyage robuste, detection d'anomalies et visualisations.
- **Jalon 3 - Baseline ML** : modele classique multilabel avec TF-IDF et regression logistique regularisee.
- **Jalon 4 - Evaluation ML** : rapport de classification, F1 micro/macro et analyse du compromis biais/variance.
- **Jalon 5 - Deep Learning fondamental** : conception d'un modele recurrent LSTM pour exploiter la sequentialite du texte.
- **Jalon 6 - Optimisation DL** : strategie contre le vanishing gradient, Adam, EarlyStopping et recherche d'hyperparametres.
- **Jalon 7 - Comparaison ML vs DL** : comparaison argumentee des scores et temps de calcul.
- **Jalon 8 - Deep Learning avance** : Transfer Learning avec un Transformer HuggingFace leger.
- **Jalon 9 - Deploiement** : dashboard Streamlit interactif pour exploiter le modele.
- **Jalon 10 - Optimisation avant/apres** : enrichissement par MyAnimeList manga, taxonomie regroupee et seuils multilabel calibres.

Les notebooks `1_eda.ipynb` a `4_deep_learning_avance.ipynb` correspondent au projet initial demande : exploration, baseline ML, LSTM et Transformer de demonstration. Les notebooks suivants sont des experiences complementaires visant a ameliorer ou tester le projet au-dela du perimetre initial :

- `5_optimisation_dataset_enrichi.ipynb` : amelioration par enrichissement des donnees, regroupement de labels et calibration des seuils;
- `6_transformer_enrichi.ipynb` : tentative de Transformer plus ambitieux sur le dataset enrichi.

Le dataset actuel contient une vraie colonne textuelle `description`, utilisee comme synopsis. La colonne `tags` sert de source de labels multilabel et est filtree pour conserver une taxonomie de genres exploitable.

## Objectifs

- Nettoyer robustement les textes : valeurs nulles, bruit HTML, URL, casse, espaces parasites et lemmatisation basique.
- Produire des statistiques descriptives utiles : volume de donnees, nombre total de mots, moyenne et mediane de mots par texte.
- Detecter les anomalies : textes vides ou trop courts apres nettoyage.
- Visualiser proprement le dataset : distribution des genres, longueur des textes et valeurs manquantes.
- Construire une baseline ML multilabel avec regularisation L2/Ridge.
- Evaluer le modele avec des metriques adaptees au multilabel, notamment F1 micro et F1 macro.
- Analyser le risque de surapprentissage ou sous-apprentissage via les scores train/test.
- Construire un modele LSTM avec embedding, gates recurrentes et sortie sigmoide multilabel.
- Comparer le LSTM a la baseline TF-IDF sur les performances et le temps d'entrainement.
- Tester un Transformer pre-entraine pour transferer des representations linguistiques generales vers NovelForge.
- Deployer une interface Streamlit avec predictions et section Transparence IA.

## Arborescence

```text
DeepLearning/
|-- data/
|   `-- data.csv
|-- notebooks/
|   |-- 1_eda.ipynb
|   |-- 2_baseline_ml.ipynb
|   |-- 3_deep_learning_fondamental.ipynb
|   |-- 4_deep_learning_avance.ipynb
|   |-- 5_optimisation_dataset_enrichi.ipynb
|   `-- 6_transformer_enrichi.ipynb
|-- scripts/
|   `-- train_enriched_baseline.py
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- visualization.py
|   |-- baseline_ml.py
|   |-- enriched_dataset.py
|   |-- dl_models.py
|   |-- transformer_model.py
|   `-- project_config.py
|-- app.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Installation

Depuis la racine du projet :

```powershell
cd "<chemin-vers-le-projet>\DeepLearning"
```

Creer un environnement virtuel si besoin :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Installer les dependances :

```powershell
pip install -r requirements.txt
```

## Lancer les notebooks

Ouvrir Jupyter :

```powershell
jupyter notebook
```

Puis executer les notebooks dans cet ordre :

```text
notebooks/1_eda.ipynb
notebooks/2_baseline_ml.ipynb
notebooks/3_deep_learning_fondamental.ipynb
notebooks/4_deep_learning_avance.ipynb
notebooks/5_optimisation_dataset_enrichi.ipynb
notebooks/6_transformer_enrichi.ipynb
```

Pour reproduire strictement le projet initial, executer seulement les notebooks 1 a 4. Les notebooks 5 et 6 sont optionnels et servent a documenter des pistes d'amelioration.

Le notebook charge automatiquement `data/data.csv`. Si le dataset porte un autre nom ou se trouve ailleurs, definir la variable d'environnement `NOVELFORGE_DATASET` :

```powershell
$env:NOVELFORGE_DATASET="C:\chemin\vers\dataset.csv"
jupyter notebook
```

Le notebook 5 detecte automatiquement si Jupyter est lance depuis la racine du projet ou depuis le dossier `notebooks/`, puis ajoute la racine au `sys.path` pour importer correctement `src`.

## Modules principaux

### `src/preprocessing.py`

Contient la logique de nettoyage et de preparation du texte :

- `TextPreprocessor` : nettoie un texte individuel ou une serie pandas.
- `drop_columns_if_present` : supprime les colonnes techniques optionnelles sans casser le pipeline.
- `parse_multilabel_cell`, `filter_labels`, `add_filtered_label_column` : normalisent les tags et les filtrent dans une taxonomie de genres.
- `clean_dataframe` : ajoute une colonne `synopsis_clean`.
- `add_text_features` : calcule `word_count`, `char_count` et `raw_char_count`.
- `detect_synopsis_anomalies` : identifie les textes vides ou inferieurs a un seuil de mots.
- `remove_synopsis_anomalies` : supprime les lignes non exploitables.

### `src/visualization.py`

Contient les fonctions de visualisation :

- `plot_top_genres` : graphique horizontal du top 15 des genres.
- `plot_synopsis_length_distribution` : histogramme de la longueur des textes.
- `plot_missing_values` : taux de valeurs manquantes par colonne.
- `count_genres` : comptage robuste des genres multilabel.

### `src/baseline_ml.py`

Contient la baseline ML du Jalon 3 :

- `BaselineModel` : pipeline scikit-learn `TfidfVectorizer` + `OneVsRestClassifier(LogisticRegression)`.
- `train()` : entraine le pipeline sur les textes nettoyes et la matrice multilabel.
- `predict()` : predit les genres sous forme de matrice binaire.
- `evaluate()` : retourne F1 micro, F1 macro, F1 weighted, Jaccard samples, Hamming loss, rapport de classification et ecarts train/test.

La regression logistique utilise une regularisation L2/Ridge. Le parametre `C` controle la force de regularisation : plus `C` est petit, plus la regularisation est forte.

### `src/dl_models.py`

Contient les briques Deep Learning du Jalon 5/6 :

- `TextVocabulary` : construit un vocabulaire word-level depuis le train uniquement.
- `TextMultilabelDataset` : convertit sequences et labels en dataset PyTorch.
- `LSTMGenreClassifier` : modele `Embedding` + `LSTM` + `Dense` final pour classification multilabel.
- `LSTMTrainingConfig` : centralise les hyperparametres du LSTM.
- `train_lstm_model()` : boucle d'entrainement PyTorch avec Adam, gradient clipping et EarlyStopping.
- `compute_pos_weight()` : pondere les classes rares dans `BCEWithLogitsLoss`.
- `find_best_threshold()` : ajuste le seuil multilabel sur validation.
- `evaluate_lstm_model()` : calcule F1 micro, F1 macro, F1 weighted, Jaccard samples et Hamming loss.

### `src/transformer_model.py`

Contient le pipeline HuggingFace du Jalon 8 :

- `TransformerConfig` : hyperparametres du modele Transformer.
- `TransformerTextDataset` : dataset PyTorch/HuggingFace pour textes et labels multilabel.
- `NovelForgeTransformer` : charge un modele pre-entraine, fine-tune legerement, predit des probabilites, evalue et sauvegarde le modele.

Le notebook avance utilise `distilbert-base-uncased`, un Transformer pre-entraine leger, sur un petit sous-echantillon pour prouver que le code tourne en local.

## Donnees attendues

Le dataset doit contenir au minimum :

- une colonne de synopsis, par exemple `synopsis`, `summary`, `description`, `overview`, `plot` ou `resume`;
- une colonne de genres ou tags, par exemple `genres`, `genre`, `tags`, `categories`, `labels` ou `target`.

Dans le dataset actuel :

- `description` est utilise comme synopsis;
- `tags` est filtre en `genre_labels`;
- `cover` est supprime via `colonnes_a_supprimer = ["cover"]`, car l'URL d'image n'est pas utile pour la baseline textuelle.

## Resultats actuels

Le notebook `2_baseline_ml.ipynb` a ete execute avec le dataset actuel :

- taille train : `55,608` lignes;
- taille test : `13,903` lignes;
- nombre de genres conserves : `34`;
- F1 micro test : environ `0.42`;
- F1 macro test : environ `0.37`.

Ces resultats sont une baseline plus realiste que la version precedente, car le modele apprend maintenant depuis les descriptions textuelles. Les scores restent moderes, ce qui est attendu pour une baseline TF-IDF lineaire sur un probleme multilabel desequilibre.

## Resultats Deep Learning actuels

Le notebook `3_deep_learning_fondamental.ipynb` a ete execute avec PyTorch sur CPU :

- lignes utilisees pour le DL : `12,000` lignes, afin de garder un temps de calcul raisonnable sans GPU;
- taille train : `8,160` lignes;
- taille validation : `1,440` lignes;
- taille test : `2,400` lignes;
- nombre de genres : `34`;
- meilleure architecture testee : LSTM bidirectionnel, embedding `128`, hidden dim `96`, dropout `0.35`;
- optimiseur : Adam;
- strategie anti-vanishing gradient : LSTM gates + gradient clipping;
- temps de recherche/entrainement : environ `380` secondes;
- F1 micro test : environ `0.19`;
- F1 macro test : environ `0.16`.

Le LSTM actuel est donc inferieur a la baseline TF-IDF. Il ameliore le rappel de nombreux genres rares, mais au prix d'une precision faible. Cette comparaison valide le Jalon 7 : le Deep Learning fondamental est plus couteux et necessite davantage de tuning pour depasser une baseline lineaire forte sur ce dataset.

## Resultats Deep Learning avance actuels

Le notebook `4_deep_learning_avance.ipynb` a ete execute avec un fine-tuning tres leger de `distilbert-base-uncased` :

- lignes utilisees : `240` lignes;
- modele : `distilbert-base-uncased`;
- entrainement : `1` epoch;
- temps d'entrainement : environ `28` secondes;
- F1 micro test : environ `0.18`.

Ce score n'est pas destine a battre la baseline : le sous-echantillon est volontairement minuscule. L'objectif du Jalon 8 est de demontrer une technologie de pointe exploitable localement : Transfer Learning, attention, fine-tuning HuggingFace, evaluation et sauvegarde du modele.

## Resultats optimisation avant/apres

Le notebook `5_optimisation_dataset_enrichi.ipynb` et le script `scripts/train_enriched_baseline.py` comparent une baseline actuelle et une baseline enrichie sur le meme test set issu du dataset NovelForge courant.

L'enrichissement utilise principalement `data/archive (1)/manga_dataset.csv`, qui contient une vraie colonne `synopsis`. La taxonomie est regroupee pour reduire le bruit :

- `Hentai`, `Ecchi`, `Erotica` et `Smut` deviennent `Adult`;
- `BL`, `GL`, `Yaoi`, `Yuri`, `Boys Love`, `Girls Love`, `Shounen-ai` et `Shoujo-ai` deviennent `BL/GL Romance`;
- `Sci-Fi` est normalise en `Sci Fi`;
- des themes MAL utiles comme `School`, `Historical`, `Isekai`, `Psychological`, `Martial Arts`, `Mecha` et `Team Sports` sont rattaches a la taxonomie enrichie.

Resultats actuels sur le test NovelForge groupe :

- baseline avant, dataset courant uniquement : F1 micro `0.485`, F1 macro `0.439`;
- baseline apres, dataset courant + MAL manga, seuil global : F1 micro `0.514`, F1 macro `0.481`;
- baseline apres, seuils par label : F1 micro `0.538`, F1 macro `0.491`.

La baseline enrichie est sauvegardee dans `models/enhanced_tfidf.joblib`, avec ses labels et seuils optimises. Les metriques sont disponibles dans `reports/enhanced_before_after_metrics.csv`.

## Transformer enrichi experimental

Le notebook `6_transformer_enrichi.ipynb` reprend le Transformer du notebook 4, mais avec un protocole plus ambitieux :

- utilisation de la taxonomie enrichie du notebook 5;
- ajout d'un echantillon de `manga_dataset.csv` dans le train;
- davantage de lignes que le notebook 4;
- calibration d'un seuil global et de seuils par label;
- evaluation sur un echantillon NovelForge pour garder un domaine de test comparable.

Ce notebook peut etre long sur CPU. Les constantes en debut de notebook permettent d'augmenter le volume si un GPU est disponible. Les artefacts sont sauvegardes separement dans `models/transformer_enriched_novelforge`, avec `models/transformer_enriched_labels.joblib`, `models/transformer_enriched_thresholds.joblib` et `models/transformer_enriched_metrics.joblib`.

Cette experience sert a verifier l'hypothese du cours : un Transformer est theoriquement plus adapte au NLP, mais son avantage pratique depend du volume de donnees, du temps d'entrainement et des ressources de calcul.

Artefacts principaux du dossier `models/` :

- `baseline_tfidf.joblib` et `baseline_labels.joblib` : baseline TF-IDF classique et ses labels;
- `enhanced_tfidf.joblib`, `enhanced_labels.joblib`, `enhanced_thresholds.joblib` : baseline enrichie, labels regroupes et seuils par label;
- `enhanced_metrics.joblib` : metriques detaillees de l'experience avant/apres;
- `lstm_novelforge.pt` et `lstm_metadata.joblib` : poids PyTorch du LSTM et metadonnees necessaires a son chargement;
- `transformer_labels.joblib` et `models/transformer_novelforge/` : labels et dossier du Transformer fine-tune.
- `transformer_enriched_labels.joblib`, `transformer_enriched_thresholds.joblib` et `models/transformer_enriched_novelforge/` : artefacts optionnels du Transformer enrichi du notebook 6.

Un fichier `.joblib` est un fichier de serialisation Python, proche d'un `pickle`, tres utilise avec scikit-learn. Il permet de sauvegarder un objet Python complet, par exemple un pipeline TF-IDF + regression logistique, une liste de labels, des seuils ou des metriques, puis de le recharger sans reentrainer le modele.

## Dashboard Streamlit

Le fichier `app.py` fournit un dashboard NovelForge :

- saisie libre d'un synopsis;
- prediction multilabel des genres;
- affichage des probabilites sous forme de graphique et barres de progression;
- comparaison cote a cote de plusieurs modeles selectionnes;
- moteur baseline TF-IDF par defaut, car c'est actuellement le plus robuste;
- moteur baseline enrichie pour comparer l'etat avant/apres l'ajout du dataset MyAnimeList manga;
- moteur LSTM PyTorch si les artefacts `models/lstm_novelforge.pt` et `models/lstm_metadata.joblib` existent;
- moteur Transformer local si un modele fine-tune est disponible dans `models/transformer_novelforge`;
- moteur Transformer enrichi si le notebook 6 a ete execute et sauvegarde `models/transformer_enriched_novelforge`;
- options avancees pour importer un CSV et reentrainer temporairement la baseline classique en session;
- onglet `Avant / Apres` affichant les metriques de comparaison entre baseline classique et baseline enrichie;
- onglet `Transparence IA` presentant les limites du projet.

Commandes pour lancer le dashboard :

```powershell
cd "<chemin-vers-le-projet>\DeepLearning"
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Si Streamlit n'est pas encore installe :

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Au premier lancement, si aucun artefact baseline classique n'existe, l'application entraine et met en cache une baseline TF-IDF locale dans `models/`.

La baseline enrichie n'est pas reentrainee depuis l'interface : elle provient du notebook 5 ou du script `scripts/train_enriched_baseline.py`. L'import CSV des options avancees ne concerne donc que la baseline classique de la session.

Pour montrer les trois etapes du projet en soutenance :

- **Baseline TF-IDF + Regression** : approche statistique rapide, efficace, mais insensible a l'ordre des mots et aux negations.
- **LSTM PyTorch** : approche sequentielle construite dans le notebook 3. Relancer la cellule de sauvegarde du notebook pour generer `models/lstm_novelforge.pt` et `models/lstm_metadata.joblib`.
- **Transformer / DistilBERT** : approche Transfer Learning avec attention globale, generee dans le notebook 4 si les artefacts Transformer sont disponibles.

Phrase piege utile pour la demonstration :

```text
Ce n'est pas une histoire d'Action, mais plutot une Romance.
```

Pour Streamlit Cloud, le dataset n'est pas versionne dans Git. Trois options sont possibles :

- placer `data/data.csv` en local pour le developpement;
- configurer un secret Streamlit `NOVELFORGE_DATASET_URL` pointant vers une URL CSV privee ou publique;
- importer le CSV depuis la barre laterale de l'application.

Exemple de secret Streamlit :

```toml
NOVELFORGE_DATASET_URL = "https://exemple.com/data.csv"
```

Les fichiers de donnees restent ignores par Git :

```powershell
git add .gitignore app.py README.md
git commit -m "Support non-versioned dataset in Streamlit"
git push
```

Si aucune de ces sources n'est disponible, l'application affiche un message d'erreur lisible au lieu d'une trace Python.

## Limites importantes

- Les modeles actuels ont ete entraines sur des synopsis en anglais. Un synopsis en francais peut etre saisi, mais les predictions seront moins fiables car le vocabulaire appris est anglophone.
- Les labels proviennent de tags editoriaux : certains representent des genres narratifs, d'autres des publics cibles, formats, tropes ou contenus adultes.
- La taxonomie enrichie regroupe volontairement certains labels pour rendre le probleme plus stable, mais cette simplification peut masquer des nuances.
- Les probabilites affichees dans Streamlit servent a classer des genres plausibles, pas a fournir une certitude absolue.

## Evaluation biais/variance

Le notebook compare les F1 train et test :

- si le F1 train est eleve mais le F1 test chute fortement, le modele surapprend;
- si les F1 train et test sont tous les deux faibles, le modele sous-apprend;
- si l'ecart train/test est raisonnable et le F1 test correct, la baseline est stable.

La regularisation L2 limite les coefficients extremes du modele TF-IDF. Elle reduit donc le risque de memoriser des tokens rares du train et aide a obtenir une meilleure generalisation.

## Prochaines etapes

- Sauvegarder un dataset nettoye intermediaire, par exemple `data/dataset_clean.csv`.
- Tester plusieurs forces de regularisation avec `C`.
- Ajuster la taxonomie de genres pour separer strictement genres, formats et tropes.
- Ameliorer le LSTM avec plus d'epochs, des seuils par label et des embeddings pre-entraines.
- Comparer ensuite le LSTM a des architectures modernes de NLP, notamment Transformers.
- Fine-tuner le Transformer sur un echantillon plus grand avec GPU.
- Calibrer les seuils de prediction par genre pour ameliorer le compromis precision/rappel.
- Evaluer les performances par genre pour mieux comprendre les classes rares.
