# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Streamlit dashboard for NovelForge genre prediction."""

from __future__ import annotations

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
BASELINE_PATH = MODELS_DIR / "baseline_tfidf.joblib"
BASELINE_LABELS_PATH = MODELS_DIR / "baseline_labels.joblib"
TRANSFORMER_DIR = MODELS_DIR / "transformer_novelforge"
TRANSFORMER_LABELS_PATH = MODELS_DIR / "transformer_labels.joblib"


def load_dataset() -> pd.DataFrame:
    data_path = PROJECT_DIR / "data" / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError("data/data.csv introuvable.")

    df_raw = pd.read_csv(data_path)
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
def load_or_train_baseline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if BASELINE_PATH.exists() and BASELINE_LABELS_PATH.exists():
        return joblib.load(BASELINE_PATH), joblib.load(BASELINE_LABELS_PATH)

    df = load_dataset()
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


def predict_with_baseline(text: str) -> pd.DataFrame:
    model, labels = load_or_train_baseline()
    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    cleaned = cleaner.clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def predict_with_transformer(text: str) -> pd.DataFrame | None:
    transformer, labels = load_transformer_if_available()
    if transformer is None:
        return None

    probabilities = transformer.predict_proba([text])[0]
    return pd.DataFrame({"genre": labels, "probability": probabilities}).sort_values("probability", ascending=False)


def render_predictions(predictions: pd.DataFrame, top_n: int) -> None:
    top_predictions = predictions.head(top_n).copy()
    st.bar_chart(top_predictions.set_index("genre")["probability"])

    for _, row in top_predictions.iterrows():
        st.progress(float(row["probability"]), text=f"{row['genre']} - {row['probability']:.2%}")


def main() -> None:
    st.set_page_config(page_title="NovelForge", page_icon="NF", layout="wide")
    st.title("NovelForge")
    st.caption("Prediction multilabel des genres d'un Light Novel / Manhwa a partir du synopsis.")

    tab_predict, tab_transparency = st.tabs(["Prediction", "Transparence IA"])

    with tab_predict:
        engine_options = ["Baseline TF-IDF"]
        if TRANSFORMER_DIR.exists() and TRANSFORMER_LABELS_PATH.exists():
            engine_options.append("Transformer local")

        engine = st.radio("Moteur de prediction", engine_options, horizontal=True)
        synopsis = st.text_area(
            "Synopsis",
            height=220,
            placeholder="Collez ici le synopsis d'un Light Novel, Manhwa ou Manhua...",
        )
        top_n = st.slider("Nombre de genres affiches", min_value=5, max_value=20, value=10)

        if st.button("Predire les genres", type="primary"):
            if len(synopsis.split()) < 5:
                st.warning("Veuillez saisir un synopsis un peu plus long.")
            else:
                if engine == "Transformer local":
                    predictions = predict_with_transformer(synopsis)
                    if predictions is None:
                        st.info("Aucun Transformer local disponible. La baseline TF-IDF est utilisee.")
                        predictions = predict_with_baseline(synopsis)
                else:
                    predictions = predict_with_baseline(synopsis)

                st.subheader("Genres les plus probables")
                render_predictions(predictions, top_n=top_n)

    with tab_transparency:
        st.header("Transparence IA")
        st.markdown(
            """
            **Limites principales**

            - Le dataset vient de tags editoriaux : certains labels sont des genres, d'autres des formats ou tropes.
            - Les genres rares sont difficiles a apprendre et peuvent etre sous-predits ou sur-predits.
            - La baseline TF-IDF ignore l'ordre global des mots, meme si elle reste robuste et rapide.
            - Le Transformer local n'est disponible que s'il a ete fine-tune et sauvegarde dans `models/transformer_novelforge`.
            - Les probabilites ne doivent pas etre lues comme des certitudes absolues : elles servent a prioriser des genres plausibles.

            **Bon usage**

            Utiliser NovelForge comme assistant d'annotation ou de recommandation, pas comme verite finale.
            Une validation humaine reste necessaire, surtout pour les synopsis ambigus ou hybrides.
            """
        )


if __name__ == "__main__":
    main()
