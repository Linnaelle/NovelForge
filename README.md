# NovelForge - Genre Prediction from Synopsis

NovelForge est un projet de Deep Learning / NLP dont l'objectif est de predire les genres d'un Light Novel ou d'un Manhwa a partir de son synopsis.

Le projet couvre actuellement :

- **Jalon 2 - EDA et preprocessing** : analyse exploratoire, nettoyage robuste, detection d'anomalies et visualisations.
- **Jalon 3 - Baseline ML** : modele classique multilabel avec TF-IDF et regression logistique regularisee.
- **Jalon 4 - Evaluation ML** : rapport de classification, F1 micro/macro et analyse du compromis biais/variance.

Le dataset actuel contient une vraie colonne textuelle `description`, utilisee comme synopsis. La colonne `tags` sert de source de labels multilabel et est filtree pour conserver une taxonomie de genres exploitable.

## Objectifs

- Nettoyer robustement les textes : valeurs nulles, bruit HTML, URL, casse, espaces parasites et lemmatisation basique.
- Produire des statistiques descriptives utiles : volume de donnees, nombre total de mots, moyenne et mediane de mots par texte.
- Detecter les anomalies : textes vides ou trop courts apres nettoyage.
- Visualiser proprement le dataset : distribution des genres, longueur des textes et valeurs manquantes.
- Construire une baseline ML multilabel avec regularisation L2/Ridge.
- Evaluer le modele avec des metriques adaptees au multilabel, notamment F1 micro et F1 macro.
- Analyser le risque de surapprentissage ou sous-apprentissage via les scores train/test.

## Arborescence

```text
DeepLearning/
|-- data/
|   `-- data.csv
|-- notebooks/
|   |-- 1_eda.ipynb
|   `-- 2_baseline_ml.ipynb
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- visualization.py
|   `-- baseline_ml.py
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
- Comparer la baseline TF-IDF a un modele Deep Learning.
- Evaluer les performances par genre pour mieux comprendre les classes rares.
