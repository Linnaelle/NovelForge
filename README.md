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
- **Jalon 9 - Deploiement** : application Streamlit interactive pour exploiter le modele.
- **Jalon 10 - Optimisation avant/apres** : enrichissement par MyAnimeList manga, taxonomie regroupee et seuils multilabel calibres.

Les notebooks `1_eda.ipynb` a `4_deep_learning_avance.ipynb` correspondent au projet initial demande : exploration, baseline ML, LSTM et Transformer de demonstration. Les notebooks suivants sont des experiences complementaires visant a ameliorer ou tester le projet au-dela du perimetre initial :

- `5_optimisation_dataset_enrichi.ipynb` : amelioration par enrichissement des donnees, regroupement de labels et calibration des seuils;
- `6_transformer_enrichi.ipynb` : tentative de Transformer plus ambitieux sur le dataset enrichi;
- `7_transformer_recherche_hyperparametres.ipynb` : recherche d'hyperparametres du Transformer enrichi.

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

## Installation

Depuis la racine du projet :

```powershell
cd ".\DeepLearning"
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
notebooks/7_transformer_recherche_hyperparametres.ipynb
```

Pour reproduire strictement le projet initial, executer seulement les notebooks 1 a 4. Les notebooks 5, 6 et 7 sont optionnels et servent a documenter des pistes d'amelioration.

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

Dans le dataset actuel :

- `description` est utilise comme synopsis;
- `tags` est filtre en `genre_labels`;
- `cover` est supprime via `colonnes_a_supprimer = ["cover"]`, car l'URL d'image n'est pas utile pour la baseline textuelle.

## Resultats actuels

Les notebooks ont ete relances le 5 juin 2026. Les scores ci-dessous sont les validations de test actuellement disponibles.

| Modele | Protocole de test | F1 micro | F1 macro | Lecture |
| --- | --- | ---: | ---: | --- |
| Baseline TF-IDF classique | Notebook 2, test NovelForge 13,903 lignes, 34 labels | `0.421` | `0.373` | Reference initiale robuste et rapide |
| LSTM PyTorch | Notebook 3, sous-echantillon CPU 2,400 lignes, 34 labels | `0.172` | `0.159` | Rappel tres fort, precision trop faible |
| Transformer demo | Notebook 4, mini test 60 lignes, 32 labels | `0.142` | `0.038` | Demonstration technique HuggingFace |
| Baseline avant regroupee | Notebook 5, meme test NovelForge groupe 10,439 lignes, 26 labels | `0.485` | `0.439` | Point de comparaison avant enrichissement |
| Baseline enrichie, seuil global | Notebook 5, meme test NovelForge groupe | `0.514` | `0.481` | Gain net avec donnees MAL manga |
| Baseline enrichie, seuils par label | Notebook 5, meme test NovelForge groupe | `0.538` | `0.491` | Meilleur modele empirique actuel |
| Transformer enrichi, seuil global | Notebook 6, test NovelForge groupe 1,200 lignes, 26 labels | `0.434` | `0.209` | Ameliore le Transformer demo, mais reste sous la baseline enrichie |
| Transformer enrichi, seuils par label | Notebook 6, test NovelForge groupe 1,200 lignes, 26 labels | `0.395` | `0.266` | Meilleur F1 macro Transformer, F1 micro plus faible |

La comparaison la plus stricte est celle du notebook 5 : les trois lignes avant/apres utilisent le meme test set. Elle montre que l'enrichissement par `manga_dataset.csv`, la taxonomie regroupee et les seuils par label ameliorent la baseline de `0.485` a `0.538` en F1 micro.

## Artefacts

Les artefacts principaux sont sauvegardes dans `models/` :

- `enhanced_tfidf.joblib`, `enhanced_labels.joblib`, `enhanced_thresholds.joblib` : meilleure baseline actuelle;
- `lstm_novelforge.pt`, `lstm_metadata.joblib` : modele LSTM et metadonnees;
- `transformer_novelforge/`, `transformer_labels.joblib` : Transformer de demonstration;
- `transformer_enriched_novelforge/`, `transformer_enriched_labels.joblib`, `transformer_enriched_thresholds.joblib` : Transformer enrichi experimental.

Le notebook 7 sauvegarde le meilleur candidat de recherche d'hyperparametres dans `transformer_enriched_novelforge/`, donc Streamlit utilise automatiquement cette version si elle remplace l'artefact precedent.

Les tableaux de metriques sont dans `reports/enhanced_before_after_metrics.csv`, `reports/transformer_enriched_metrics.csv` et, apres le notebook 7, `reports/transformer_hparam_search_metrics.csv`.

## Modeles sur Hugging Face

Les gros artefacts ne sont pas versionnes dans Git. L'application les telecharge depuis le repo Hugging Face `Linnaelle/NovelForge` quand ils manquent dans `models/`.

Convention d'upload recommandee : envoyer le contenu de `models/` a la racine du repo HF.

```powershell
hf auth login
hf upload Linnaelle/NovelForge models/ .
```

Pour Streamlit Cloud ou une CI, configurer ces secrets si besoin :

- `NOVELFORGE_MODEL_REPO=Linnaelle/NovelForge`
- `HF_TOKEN=<token Hugging Face>` si le repo est prive

Les fichiers attendus cote HF sont par exemple `enhanced_tfidf.joblib`, `enhanced_labels.joblib`, `lstm_novelforge.pt`, `transformer_novelforge/` et `transformer_enriched_novelforge/`.

## Application Streamlit

Le fichier `app.py` fournit une interface utilisateur pour :

- saisir un synopsis et comparer les predictions de plusieurs modeles;
- visualiser les genres les plus probables;
- verifier les scores de test de chaque modele dans l'onglet `Comparaison`;
- distinguer les resultats directement comparables des experiences exploratoires;
- expliquer les limites du systeme dans l'onglet `Transparence IA`.

Commandes pour lancer l'application :

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

Au premier lancement, si aucun artefact baseline classique n'existe, l'application entraine et met en cache une baseline TF-IDF locale dans `models/`. La baseline enrichie n'est pas reentrainee depuis l'interface : elle provient du notebook 5 ou du script `scripts/train_enriched_baseline.py`.

Pour Streamlit Cloud, le dataset n'est pas versionne dans Git. Trois options sont possibles :

- placer `data/data.csv` en local pour le developpement;
- configurer un secret Streamlit `NOVELFORGE_DATASET_URL` pointant vers une URL CSV privee ou publique;
- importer le CSV depuis la barre laterale de l'application.

## Limites importantes

- Les modeles actuels ont ete entraines sur des synopsis en anglais. Un synopsis en francais peut etre saisi, mais les predictions seront moins fiables car le vocabulaire appris est anglophone.
- Les labels proviennent de tags editoriaux : certains representent des genres narratifs, d'autres des publics cibles, formats, tropes ou contenus adultes.
- La taxonomie enrichie regroupe volontairement certains labels pour rendre le probleme plus stable, mais cette simplification peut masquer des nuances.
- Les probabilites affichees dans Streamlit servent a classer des genres plausibles, pas a fournir une certitude absolue.
