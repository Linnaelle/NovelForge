# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Streamlit dashboard for NovelForge genre prediction."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_DIR = Path(__file__).resolve().parent

from src.baseline_ml import BaselineModel
from src.preprocessing import (
    TextPreprocessor,
    add_filtered_label_column,
    clean_dataframe,
    drop_columns_if_present,
    infer_column,
    parse_multilabel_cell,
    remove_synopsis_anomalies,
)
from src.project_config import GENRE_VOCABULARY


MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
BASELINE_PATH = MODELS_DIR / "baseline_tfidf.joblib"
BASELINE_LABELS_PATH = MODELS_DIR / "baseline_labels.joblib"
ENHANCED_PATH = MODELS_DIR / "enhanced_tfidf.joblib"
ENHANCED_LABELS_PATH = MODELS_DIR / "enhanced_labels.joblib"
ENHANCED_THRESHOLDS_PATH = MODELS_DIR / "enhanced_thresholds.joblib"
ENHANCED_METRICS_PATH = REPORTS_DIR / "enhanced_before_after_metrics.csv"
LSTM_PATH = MODELS_DIR / "lstm_novelforge.pt"
LSTM_METADATA_PATH = MODELS_DIR / "lstm_metadata.joblib"
TRANSFORMER_DIR = MODELS_DIR / "transformer_novelforge"
TRANSFORMER_LABELS_PATH = MODELS_DIR / "transformer_labels.joblib"
ENHANCED_TRANSFORMER_DIR = MODELS_DIR / "transformer_enriched_novelforge"
ENHANCED_TRANSFORMER_LABELS_PATH = MODELS_DIR / "transformer_enriched_labels.joblib"


def find_dataset_path() -> Path | None:
    """Return the first available local dataset path."""
    candidates = [
        PROJECT_DIR / "data" / "data.csv",
        PROJECT_DIR / "data.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def get_configured_dataset_url() -> str | None:
    """Read a private dataset URL from environment variables or Streamlit secrets."""
    env_url = os.getenv("NOVELFORGE_DATASET_URL")
    if env_url:
        return env_url

    try:
        secret_url = st.secrets.get("NOVELFORGE_DATASET_URL") or st.secrets.get("dataset_url")
    except Exception:
        secret_url = None

    return str(secret_url) if secret_url else None


def load_dataset(uploaded_dataset: bytes | None = None, dataset_url: str | None = None) -> pd.DataFrame:
    """Load the dataset without requiring it to be versioned in Git."""
    if uploaded_dataset is not None:
        df_raw = pd.read_csv(BytesIO(uploaded_dataset))
    else:
        data_path = find_dataset_path()
        if data_path is not None:
            df_raw = pd.read_csv(data_path)
        elif dataset_url:
            df_raw = pd.read_csv(dataset_url)
        else:
            raise FileNotFoundError(
                "Dataset introuvable. Le dataset n'est pas versionne dans Git : "
                "ajoute `data/data.csv` en local, configure le secret Streamlit "
                "`NOVELFORGE_DATASET_URL`, ou importe un CSV depuis les options avancees."
            )

    df_raw = drop_columns_if_present(df_raw, ["cover"])

    text_column = infer_column(df_raw.columns, ["synopsis", "summary", "description", "overview", "plot", "resume"])
    label_column = infer_column(df_raw.columns, ["genres", "genre", "tags", "categories", "labels", "target"])

    df_raw = add_filtered_label_column(df_raw, label_column, "genre_labels", GENRE_VOCABULARY)
    preprocessor = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    df = clean_dataframe(df_raw, text_column=text_column, clean_column="synopsis_clean", preprocessor=preprocessor)
    df = remove_synopsis_anomalies(df, clean_column="synopsis_clean", min_words=5)
    df = df[df["genre_labels"].str.len().gt(0)].copy()
    return df


@st.cache_resource(show_spinner="Chargement ou entrainement de la baseline TF-IDF...")
def load_or_train_baseline(uploaded_dataset: bytes | None = None, dataset_url: str | None = None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if uploaded_dataset is None and dataset_url is None and BASELINE_PATH.exists() and BASELINE_LABELS_PATH.exists():
        return joblib.load(BASELINE_PATH), joblib.load(BASELINE_LABELS_PATH)

    df = load_dataset(uploaded_dataset=uploaded_dataset, dataset_url=dataset_url)
    df_model = df[["synopsis_clean", "genre_labels"]].copy()
    df_model["genre_labels"] = df_model["genre_labels"].apply(parse_multilabel_cell)
    df_model = df_model[df_model["genre_labels"].str.len().gt(0)].copy()

    max_rows = 20_000
    if len(df_model) > max_rows:
        df_model = df_model.sample(n=max_rows, random_state=42).copy()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_model["genre_labels"])
    x = df_model["synopsis_clean"].fillna("").astype(str)

    model = BaselineModel(max_features=30_000, ngram_range=(1, 2), min_df=2, max_iter=1_000)
    model.train(x, y)

    joblib.dump(model, BASELINE_PATH)
    joblib.dump(list(mlb.classes_), BASELINE_LABELS_PATH)
    return model, list(mlb.classes_)


@st.cache_resource(show_spinner="Chargement de la baseline enrichie...")
def load_enhanced_baseline_if_available():
    if not ENHANCED_PATH.exists() or not ENHANCED_LABELS_PATH.exists():
        return None, None, None

    model = joblib.load(ENHANCED_PATH)
    labels = joblib.load(ENHANCED_LABELS_PATH)
    thresholds = joblib.load(ENHANCED_THRESHOLDS_PATH) if ENHANCED_THRESHOLDS_PATH.exists() else None
    return model, labels, thresholds


@st.cache_data(show_spinner=False)
def load_enhanced_metrics() -> pd.DataFrame | None:
    if not ENHANCED_METRICS_PATH.exists():
        return None

    return pd.read_csv(ENHANCED_METRICS_PATH)


@st.cache_resource(show_spinner="Chargement du Transformer local...")
def load_transformer_if_available():
    if not TRANSFORMER_DIR.exists() or not TRANSFORMER_LABELS_PATH.exists():
        return None, None

    from src.transformer_model import NovelForgeTransformer, TransformerConfig

    labels = joblib.load(TRANSFORMER_LABELS_PATH)
    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    config = TransformerConfig(model_name=str(TRANSFORMER_DIR), output_dir=str(TRANSFORMER_DIR))
    return NovelForgeTransformer.load(TRANSFORMER_DIR, id2label=id2label, label2id=label2id, config=config), labels


@st.cache_resource(show_spinner="Chargement du Transformer enrichi...")
def load_enhanced_transformer_if_available():
    if not ENHANCED_TRANSFORMER_DIR.exists() or not ENHANCED_TRANSFORMER_LABELS_PATH.exists():
        return None, None

    from src.transformer_model import NovelForgeTransformer, TransformerConfig

    labels = joblib.load(ENHANCED_TRANSFORMER_LABELS_PATH)
    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    config = TransformerConfig(model_name=str(ENHANCED_TRANSFORMER_DIR), output_dir=str(ENHANCED_TRANSFORMER_DIR))
    return NovelForgeTransformer.load(
        ENHANCED_TRANSFORMER_DIR,
        id2label=id2label,
        label2id=label2id,
        config=config,
    ), labels


@st.cache_resource(show_spinner="Chargement du LSTM local...")
def load_lstm_if_available():
    if not LSTM_PATH.exists() or not LSTM_METADATA_PATH.exists():
        return None, None, None, None

    import torch

    from src.dl_models import LSTMGenreClassifier, LSTMTrainingConfig, TextVocabulary, get_device

    metadata = joblib.load(LSTM_METADATA_PATH)
    labels = metadata["labels"]
    config = LSTMTrainingConfig(**metadata["config"])
    config.threshold = float(metadata.get("threshold", config.threshold))

    vocabulary = TextVocabulary(
        max_vocab_size=config.max_vocab_size,
        min_freq=config.min_freq,
    )
    vocabulary.token_to_id = metadata["token_to_id"]
    vocabulary.id_to_token = metadata["id_to_token"]

    model = LSTMGenreClassifier(
        vocab_size=vocabulary.size,
        num_labels=len(labels),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )
    device = get_device()
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, vocabulary, labels, config


def predict_with_baseline(
    text: str,
    uploaded_dataset: bytes | None = None,
    dataset_url: str | None = None,
) -> pd.DataFrame:
    model, labels = load_or_train_baseline(uploaded_dataset=uploaded_dataset, dataset_url=dataset_url)
    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    cleaned = cleaner.clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def predict_with_enhanced_baseline(text: str) -> pd.DataFrame | None:
    model, labels, thresholds = load_enhanced_baseline_if_available()
    if model is None:
        return None

    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    cleaned = cleaner.clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    predictions = pd.DataFrame({"genre": labels, "probability": probabilities})

    if thresholds is not None:
        predictions["threshold"] = thresholds
        predictions["selected"] = predictions["probability"].ge(predictions["threshold"])

    return predictions.sort_values("probability", ascending=False)


def predict_with_lstm(text: str) -> pd.DataFrame | None:
    lstm_model, vocabulary, labels, config = load_lstm_if_available()
    if lstm_model is None:
        return None

    import torch

    from src.dl_models import get_device

    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    cleaned = cleaner.clean_text(text)
    sequence = vocabulary.transform([cleaned], max_length=config.max_length)
    input_ids = torch.as_tensor(sequence, dtype=torch.long, device=get_device())

    with torch.no_grad():
        probabilities = torch.sigmoid(lstm_model(input_ids)).cpu().numpy()[0]

    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def predict_with_transformer(text: str) -> pd.DataFrame | None:
    transformer, labels = load_transformer_if_available()
    if transformer is None:
        return None

    probabilities = transformer.predict_proba([text])[0]
    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def predict_with_enhanced_transformer(text: str) -> pd.DataFrame | None:
    transformer, labels = load_enhanced_transformer_if_available()
    if transformer is None:
        return None

    probabilities = transformer.predict_proba([text])[0]
    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def render_predictions(predictions: pd.DataFrame, top_n: int) -> None:
    top_predictions = predictions.head(top_n).copy()
    st.bar_chart(top_predictions.set_index("genre")["probability"])

    for _, row in top_predictions.iterrows():
        st.progress(float(row["probability"]), text=f"{row['genre']} - {row['probability']:.2%}")

    if "selected" in predictions.columns:
        selected = predictions[predictions["selected"]].sort_values("probability", ascending=False)
        if not selected.empty:
            st.caption("Genres retenus par les seuils optimises : " + ", ".join(selected["genre"].head(top_n)))


def main() -> None:
    st.set_page_config(page_title="NovelForge", page_icon="NF", layout="wide")
    st.title("NovelForge")
    st.caption("Prediction multilabel des genres d'un Light Novel / Manhwa a partir du synopsis.")

    dataset_url = get_configured_dataset_url()
    uploaded_dataset_bytes = None

    with st.sidebar:
        st.header("Donnees")
        if find_dataset_path() is not None:
            st.success("Dataset local detecte.")
        elif dataset_url:
            st.info("Dataset charge via secret Streamlit.")
        else:
            st.warning("Aucun dataset disponible pour entrainer la baseline.")

        with st.expander("Options avancees", expanded=False):
            st.caption(
                "L'import CSV sert uniquement a reentrainer temporairement la baseline TF-IDF classique. "
                "La baseline enrichie sauvegardee n'est pas modifiee."
            )
            uploaded_dataset = st.file_uploader(
                "Dataset CSV optionnel",
                type=["csv"],
                help="Le CSV doit contenir une colonne de texte et une colonne de genres/tags.",
            )
            if uploaded_dataset is not None:
                uploaded_dataset_bytes = uploaded_dataset.getvalue()
                st.success("Dataset charge pour cette session.")

        st.header("Modeles")
        st.success("Baseline TF-IDF disponible")
        if LSTM_PATH.exists() and LSTM_METADATA_PATH.exists():
            st.success("LSTM PyTorch disponible")
        else:
            st.info("LSTM absent : relancer la cellule de sauvegarde du notebook 3.")
        if TRANSFORMER_DIR.exists() and TRANSFORMER_LABELS_PATH.exists():
            st.success("Transformer local disponible")
        else:
            st.info("Transformer absent : relancer le notebook 4 si besoin.")
        if ENHANCED_TRANSFORMER_DIR.exists() and ENHANCED_TRANSFORMER_LABELS_PATH.exists():
            st.success("Transformer enrichi disponible")
        else:
            st.info("Transformer enrichi absent : relancer le notebook 6 si besoin.")
        if ENHANCED_PATH.exists() and ENHANCED_LABELS_PATH.exists():
            st.success("Baseline enrichie disponible")
        else:
            st.info("Baseline enrichie absente : lancer `scripts/train_enriched_baseline.py`.")

    tab_predict, tab_compare, tab_transparency = st.tabs(["Prediction", "Avant / Apres", "Transparence IA"])

    with tab_predict:
        with st.expander("Guide de demonstration des 3 approches", expanded=False):
            st.markdown(
                """
                **1. Baseline TF-IDF + Regression** : approche statistique tres rapide. Elle lit surtout les mots presents,
                mais ignore l'ordre et la negation.

                **2. LSTM PyTorch** : approche sequentielle construite de A a Z. Elle lit les tokens dans l'ordre,
                mais reste limitee par la taille d'entrainement et son F1 micro plus faible.

                **3. Transformer / DistilBERT** : approche Transfer Learning avec attention globale. Elle sert a montrer
                pourquoi l'etat de l'art est mieux arme pour relier les mots importants dans une phrase.
                """
            )
            st.code("Ce n'est pas une histoire d'Action, mais plutot une Romance.", language="text")

        engine_options = ["Baseline TF-IDF"]
        if ENHANCED_PATH.exists() and ENHANCED_LABELS_PATH.exists():
            engine_options.append("Baseline enrichie")
        if LSTM_PATH.exists() and LSTM_METADATA_PATH.exists():
            engine_options.append("LSTM PyTorch")
        if TRANSFORMER_DIR.exists() and TRANSFORMER_LABELS_PATH.exists():
            engine_options.append("Transformer local")
        if ENHANCED_TRANSFORMER_DIR.exists() and ENHANCED_TRANSFORMER_LABELS_PATH.exists():
            engine_options.append("Transformer enrichi")

        selected_engines = st.multiselect(
            "Modeles a comparer",
            options=engine_options,
            default=["Baseline TF-IDF"],
        )
        synopsis = st.text_area(
            "Synopsis",
            height=220,
            placeholder="Collez ici le synopsis d'un Light Novel, Manhwa ou Manhua...",
        )
        top_n = st.slider("Nombre de genres affiches", min_value=5, max_value=20, value=10)

        if st.button("Predire les genres", type="primary"):
            if len(synopsis.split()) < 5:
                st.warning("Veuillez saisir un synopsis un peu plus long.")
            elif not selected_engines:
                st.warning("Veuillez selectionner au moins un modele.")
            else:
                columns = st.columns(len(selected_engines))

                for column, engine in zip(columns, selected_engines):
                    with column:
                        st.subheader(engine)

                        try:
                            if engine == "Transformer local":
                                predictions = predict_with_transformer(synopsis)
                                if predictions is None:
                                    st.info("Aucun Transformer local disponible.")
                                    continue
                            elif engine == "Transformer enrichi":
                                predictions = predict_with_enhanced_transformer(synopsis)
                                if predictions is None:
                                    st.info("Aucun Transformer enrichi disponible.")
                                    continue
                            elif engine == "Baseline enrichie":
                                predictions = predict_with_enhanced_baseline(synopsis)
                                if predictions is None:
                                    st.info("Aucune baseline enrichie disponible.")
                                    continue
                            elif engine == "LSTM PyTorch":
                                predictions = predict_with_lstm(synopsis)
                                if predictions is None:
                                    st.info("Aucun LSTM local disponible. Relance le notebook 3 pour sauvegarder ses artefacts.")
                                    continue
                            else:
                                predictions = predict_with_baseline(
                                    synopsis,
                                    uploaded_dataset=uploaded_dataset_bytes,
                                    dataset_url=dataset_url,
                                )

                            render_predictions(predictions, top_n=top_n)
                        except FileNotFoundError as error:
                            st.error(str(error))

    with tab_compare:
        st.header("Avant / Apres")
        metrics = load_enhanced_metrics()
        if metrics is None:
            st.info("Aucune metrique enrichie trouvee. Lance `scripts/train_enriched_baseline.py` pour generer la comparaison.")
        else:
            display_columns = [
                "model",
                "threshold_strategy",
                "train_rows",
                "external_rows",
                "f1_micro",
                "f1_macro",
                "f1_weighted",
                "jaccard_samples",
                "hamming_loss",
            ]
            st.dataframe(metrics[display_columns], use_container_width=True, hide_index=True)

            chart = metrics.set_index("model")[["f1_micro", "f1_macro", "f1_weighted"]]
            chart.index = metrics["model"] + " / " + metrics["threshold_strategy"]
            st.bar_chart(chart)

            best_row = metrics.sort_values("f1_micro", ascending=False).iloc[0]
            st.success(
                "Meilleur resultat actuel : "
                f"{best_row['model']} ({best_row['threshold_strategy']}) "
                f"avec F1 micro {best_row['f1_micro']:.3f}."
            )

    with tab_transparency:
        st.header("Transparence IA")
        st.markdown(
            """
            **Limites principales**

            - Le dataset vient de tags editoriaux : certains labels sont des genres, d'autres des formats ou tropes.
            - Les genres rares sont difficiles a apprendre et peuvent etre sous-predits ou sur-predits.
            - La baseline TF-IDF ignore l'ordre global des mots, meme si elle reste robuste et rapide.
            - Le LSTM lit les sequences, mais son entrainement local limite peut produire des probabilites hesitantes.
            - Le Transformer local n'est disponible que s'il a ete fine-tune et sauvegarde dans `models/transformer_novelforge`.
            - La baseline enrichie utilise une taxonomie regroupee : certains labels adultes ou BL/GL sont volontairement fusionnes.
            - Les modeles ont ete entraines sur des synopsis en anglais : les textes en francais sont acceptes techniquement, mais les predictions sont moins fiables.
            - L'import CSV des options avancees ne modifie que la baseline classique de la session, pas le modele enrichi sauvegarde.
            - Les probabilites ne doivent pas etre lues comme des certitudes absolues : elles servent a prioriser des genres plausibles.

            **Bon usage**

            Utiliser NovelForge comme assistant d'annotation ou de recommandation, pas comme verite finale.
            Une validation humaine reste necessaire, surtout pour les synopsis ambigus ou hybrides.
            """
        )


if __name__ == "__main__":
    main()
