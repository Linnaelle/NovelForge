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
|   `-- 4_deep_learning_avance.ipynb
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- visualization.py
|   |-- baseline_ml.py
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
```

Le notebook charge automatiquement `data/data.csv`. Si le dataset porte un autre nom ou se trouve ailleurs, definir la variable d'environnement `NOVELFORGE_DATASET` :

```powershell
$env:NOVELFORGE_DATASET="C:\chemin\vers\dataset.csv"
jupyter notebook
```

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

## Dashboard Streamlit

Le fichier `app.py` fournit un dashboard NovelForge :

- saisie libre d'un synopsis;
- prediction multilabel des genres;
- affichage des probabilites sous forme de graphique et barres de progression;
- moteur baseline TF-IDF par defaut, car c'est actuellement le plus robuste;
- moteur Transformer local si un modele fine-tune est disponible dans `models/transformer_novelforge`;
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

Au premier lancement, si aucun artefact baseline n'existe, l'application entraine et met en cache une baseline TF-IDF locale dans `models/`.

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
