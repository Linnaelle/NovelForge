# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Streamlit app for NovelForge genre prediction."""

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
from src.model_store import DEFAULT_MODEL_REPO_ID, ensure_model_dir, ensure_model_file
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
ENHANCED_TRANSFORMER_METRICS_PATH = REPORTS_DIR / "transformer_enriched_metrics.csv"
LSTM_PATH = MODELS_DIR / "lstm_novelforge.pt"
LSTM_METADATA_PATH = MODELS_DIR / "lstm_metadata.joblib"
TRANSFORMER_DIR = MODELS_DIR / "transformer_novelforge"
TRANSFORMER_LABELS_PATH = MODELS_DIR / "transformer_labels.joblib"
ENHANCED_TRANSFORMER_DIR = MODELS_DIR / "transformer_enriched_novelforge"
ENHANCED_TRANSFORMER_LABELS_PATH = MODELS_DIR / "transformer_enriched_labels.joblib"


def get_configured_model_repo() -> str:
    """Read the Hugging Face model repository from env vars or Streamlit secrets."""
    env_repo = os.getenv("NOVELFORGE_MODEL_REPO")
    if env_repo:
        return env_repo

    try:
        secret_repo = st.secrets.get("NOVELFORGE_MODEL_REPO") or st.secrets.get("model_repo")
    except Exception:
        secret_repo = None

    return str(secret_repo) if secret_repo else DEFAULT_MODEL_REPO_ID


def get_configured_hf_token() -> str | None:
    """Read an optional Hugging Face token from env vars or Streamlit secrets."""
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if env_token:
        return env_token

    try:
        secret_token = st.secrets.get("HF_TOKEN") or st.secrets.get("hf_token")
    except Exception:
        secret_token = None

    return str(secret_token) if secret_token else None


def ensure_hf_file(filename: str, target_path: Path) -> Path | None:
    """Download a single artifact from the configured HF repo when missing."""
    return ensure_model_file(
        filename,
        target_path,
        repo_id=get_configured_model_repo(),
        token=get_configured_hf_token(),
    )


def ensure_hf_dir(dirname: str, target_dir: Path) -> Path | None:
    """Download an artifact directory from the configured HF repo when missing."""
    return ensure_model_dir(
        dirname,
        target_dir,
        repo_id=get_configured_model_repo(),
        token=get_configured_hf_token(),
    )


def has_local_or_remote_models() -> bool:
    """Return whether remote model loading is configured for the app."""
    return bool(get_configured_model_repo())


MODEL_TEST_RESULTS = [
    {
        "model": "Baseline TF-IDF classique",
        "test_protocol": "Notebook 2 - split aleatoire NovelForge",
        "test_rows": 13_903,
        "labels": 34,
        "threshold_strategy": "global 0.50",
        "f1_micro": 0.4213,
        "f1_macro": 0.3734,
        "jaccard_samples": 0.2849,
        "hamming_loss": 0.1393,
        "comparison_scope": "Reference initiale",
    },
    {
        "model": "LSTM PyTorch",
        "test_protocol": "Notebook 3 - sous-echantillon CPU NovelForge",
        "test_rows": 2_400,
        "labels": 34,
        "threshold_strategy": "global 0.45",
        "f1_micro": 0.1718,
        "f1_macro": 0.1588,
        "jaccard_samples": 0.0931,
        "hamming_loss": 0.8670,
        "comparison_scope": "Validation pedagogique",
    },
    {
        "model": "Transformer demo",
        "test_protocol": "Notebook 4 - mini test NovelForge",
        "test_rows": 60,
        "labels": 32,
        "threshold_strategy": "global 0.50",
        "f1_micro": 0.1424,
        "f1_macro": 0.0375,
        "jaccard_samples": 0.0788,
        "hamming_loss": 0.1443,
        "comparison_scope": "Demo technique",
    },
]


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

    if uploaded_dataset is None and dataset_url is None:
        ensure_hf_file("baseline_tfidf.joblib", BASELINE_PATH)
        ensure_hf_file("baseline_labels.joblib", BASELINE_LABELS_PATH)

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
    ensure_hf_file("enhanced_tfidf.joblib", ENHANCED_PATH)
    ensure_hf_file("enhanced_labels.joblib", ENHANCED_LABELS_PATH)
    ensure_hf_file("enhanced_thresholds.joblib", ENHANCED_THRESHOLDS_PATH)

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


@st.cache_data(show_spinner=False)
def load_enhanced_transformer_metrics() -> pd.DataFrame | None:
    if not ENHANCED_TRANSFORMER_METRICS_PATH.exists():
        return None

    return pd.read_csv(ENHANCED_TRANSFORMER_METRICS_PATH)


def build_test_results_table() -> pd.DataFrame:
    rows = list(MODEL_TEST_RESULTS)

    enhanced_metrics = load_enhanced_metrics()
    if enhanced_metrics is not None:
        for _, row in enhanced_metrics.iterrows():
            model_name = "Baseline enrichie" if str(row["model"]).startswith("after") else "Baseline avant regroupee"
            rows.append(
                {
                    "model": model_name,
                    "test_protocol": "Notebook 5 - meme test NovelForge groupe",
                    "test_rows": 10_439,
                    "labels": 26,
                    "threshold_strategy": row["threshold_strategy"],
                    "f1_micro": row["f1_micro"],
                    "f1_macro": row["f1_macro"],
                    "jaccard_samples": row["jaccard_samples"],
                    "hamming_loss": row["hamming_loss"],
                    "comparison_scope": "Avant/apres comparable",
                }
            )

    transformer_metrics = load_enhanced_transformer_metrics()
    if transformer_metrics is not None:
        for _, row in transformer_metrics.iterrows():
            rows.append(
                {
                    "model": "Transformer enrichi",
                    "test_protocol": "Notebook 6 - echantillon NovelForge groupe",
                    "test_rows": int(row["test_rows"]),
                    "labels": 26,
                    "threshold_strategy": row["threshold_strategy"],
                    "f1_micro": row["f1_micro"],
                    "f1_macro": row["f1_macro"],
                    "jaccard_samples": row["jaccard_samples"],
                    "hamming_loss": row["hamming_loss"],
                    "comparison_scope": "Exploration avancee",
                }
            )

    results = pd.DataFrame(rows)
    return results.sort_values(["f1_micro", "f1_macro"], ascending=False).reset_index(drop=True)


@st.cache_resource(show_spinner="Chargement du Transformer local...")
def load_transformer_if_available():
    ensure_hf_dir("transformer_novelforge", TRANSFORMER_DIR)
    ensure_hf_file("transformer_labels.joblib", TRANSFORMER_LABELS_PATH)

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
    ensure_hf_dir("transformer_enriched_novelforge", ENHANCED_TRANSFORMER_DIR)
    ensure_hf_file("transformer_enriched_labels.joblib", ENHANCED_TRANSFORMER_LABELS_PATH)
    ensure_hf_file("transformer_enriched_thresholds.joblib", MODELS_DIR / "transformer_enriched_thresholds.joblib")

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
    ensure_hf_file("lstm_novelforge.pt", LSTM_PATH)
    ensure_hf_file("lstm_metadata.joblib", LSTM_METADATA_PATH)

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
    
    tags_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem;">'
    for _, row in top_predictions.iterrows():
        prob = float(row["probability"])
        
        if prob > 0.50:
            bg_color = "rgba(255, 75, 75, 0.15)"
            border_color = "rgba(255, 75, 75, 0.6)"
        elif prob > 0.25:
            bg_color = "rgba(255, 164, 164, 0.1)"
            border_color = "rgba(255, 164, 164, 0.5)"
        else:
            bg_color = "rgba(128, 128, 128, 0.05)"
            border_color = "rgba(128, 128, 128, 0.3)"
            
        tags_html += f"""
        <div style="padding: 0.3rem 0.8rem; border-radius: 20px; 
                    background-color: {bg_color}; border: 1px solid {border_color};
                    color: var(--text-color); font-size: 0.9rem; font-weight: 500;
                    display: flex; align-items: center; gap: 6px;">
            {row['genre']} 
            <span style="opacity: 0.5; font-size: 0.8rem;">{prob:.0%}</span>
        </div>
        """
    tags_html += "</div>"
    
    st.html(tags_html)

    with st.expander("📊 Voir le détail analytique"):
        for _, row in top_predictions.iterrows():
            st.progress(float(row["probability"]), text=f"{row['genre']}")

        if "selected" in predictions.columns:
            selected = predictions[predictions["selected"]].sort_values("probability", ascending=False)
            if not selected.empty:
                st.caption("✨ Genres validés par les seuils : " + ", ".join(selected["genre"].head(top_n)))


def main() -> None:
    st.set_page_config(page_title="NovelForge", page_icon="✨", layout="centered")
    
    st.markdown(
        """
        <style>
        /* Aérer le conteneur principal */
        .block-container { padding-top: 3rem; max-width: 850px; }
        
        /* Styliser le titre pour qu'il s'adapte au thème de l'utilisateur */
        .main-title { 
            text-align: center; 
            font-size: 3.5rem; 
            font-weight: 800; 
            color: var(--text-color);
            margin-bottom: 0.5rem;
        }
        .subtitle { 
            text-align: center; 
            font-size: 1.1rem; 
            opacity: 0.7; 
            color: var(--text-color);
            margin-bottom: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div class='main-title'>NovelForge ✨</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>L'IA qui découvre les genres littéraires cachés dans votre synopsis.</div>", unsafe_allow_html=True)

    dataset_url = get_configured_dataset_url()
    model_repo = get_configured_model_repo()
    remote_models_enabled = has_local_or_remote_models()
    uploaded_dataset_bytes = None

    with st.sidebar:
        st.header("⚙️ Réglages")
        top_n = st.slider("Nombre de genres à afficher", min_value=3, max_value=15, value=8)
        
        with st.expander("🛠️ Mode Développeur (Statut des modèles)", expanded=False):
            if find_dataset_path() is not None:
                st.success("Dataset local : OK")
            st.info(f"Repo HF modeles : {model_repo}")
            
            st.markdown("**Moteurs de prédiction :**")
            if BASELINE_PATH.exists() and BASELINE_LABELS_PATH.exists(): st.write("Baseline TF-IDF locale : OK")
            if ENHANCED_PATH.exists() and ENHANCED_LABELS_PATH.exists(): st.write("Baseline enrichie locale : OK")
            if LSTM_PATH.exists() and LSTM_METADATA_PATH.exists(): st.write("LSTM PyTorch local : OK")
            if TRANSFORMER_DIR.exists(): st.write("✅ Transformer local")
            if ENHANCED_TRANSFORMER_DIR.exists(): st.write("✅ Transformer enrichi")
            if remote_models_enabled: st.write("HF actif : les modeles manquants seront telecharges a la demande.")

            uploaded_dataset = st.file_uploader("Forcer un CSV local", type=["csv"])
            if uploaded_dataset is not None:
                uploaded_dataset_bytes = uploaded_dataset.getvalue()

    tab_predict, tab_compare, tab_about = st.tabs(["📝 Analyseur", "🔬 Comparatif IA", "ℹ️ Comment ça marche ?"])

    with tab_predict:
        synopsis = st.text_area(
            "Collez votre résumé :",
            height=200,
            placeholder="Dans un monde où la magie a disparu, un jeune forgeron découvre une épée ancienne qui chuchote dans son esprit...",
            label_visibility="visible"
        )
        
        engine_options = []
        if remote_models_enabled or ENHANCED_PATH.exists(): engine_options.append("Baseline enrichie (Recommandé)")
        engine_options.append("Baseline TF-IDF (Rapide)")
        if remote_models_enabled or ENHANCED_TRANSFORMER_DIR.exists(): engine_options.append("Transformer enrichi")
        if remote_models_enabled or TRANSFORMER_DIR.exists(): engine_options.append("Transformer local")
        if remote_models_enabled or LSTM_PATH.exists(): engine_options.append("LSTM PyTorch (Expérimental)")

        selected_engines = st.multiselect(
            "Quelles IA voulez-vous consulter ?",
            options=engine_options,
            default=[engine_options[0]] if engine_options else ["Baseline TF-IDF (Rapide)"],
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("✨ Analyser le synopsis", type="primary", width='stretch')

        if analyze_button:
            if len(synopsis.split()) < 5:
                st.warning("👋 Le synopsis est un peu trop court pour que l'IA puisse l'analyser correctement.")
            elif not selected_engines:
                st.warning("Veuillez sélectionner au moins un modèle d'IA.")
            else:
                st.markdown("---")
                columns = st.columns(len(selected_engines))

                for column, engine in zip(columns, selected_engines):
                    with column:
                        clean_name = engine.replace(" (Recommandé)", "").replace(" (Rapide)", "").replace(" (Expérimental)", "")
                        st.markdown(f"### 🤖 {clean_name}")

                        try:
                            if "Transformer local" in engine:
                                predictions = predict_with_transformer(synopsis)
                            elif "Transformer enrichi" in engine:
                                predictions = predict_with_enhanced_transformer(synopsis)
                            elif "Baseline enrichie" in engine:
                                predictions = predict_with_enhanced_baseline(synopsis)
                            elif "LSTM" in engine:
                                predictions = predict_with_lstm(synopsis)
                            else:
                                predictions = predict_with_baseline(synopsis, uploaded_dataset=uploaded_dataset_bytes, dataset_url=dataset_url)

                            if predictions is not None:
                                render_predictions(predictions, top_n=top_n)
                            else:
                                st.error("Modèle introuvable sur le serveur.")
                                
                        except Exception as error:
                            st.error(f"Une erreur est survenue : {str(error)}")

    with tab_compare:
        st.header("Performances des modèles")
        st.markdown("Cet espace est dédié à l'évaluation technique des différentes architectures entraînées pour ce projet.")

        test_results = build_test_results_table()
        if not test_results.empty:
            best = test_results.iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Meilleur modèle", str(best["model"]))
            metric_cols[1].metric("F1 micro test", f"{best['f1_micro']:.3f}")
            metric_cols[2].metric("F1 macro test", f"{best['f1_macro']:.3f}")
            metric_cols[3].metric("Test", f"{int(best['test_rows']):,} lignes")
            
            st.dataframe(test_results, width='stretch', hide_index=True)

    with tab_about:
        st.header("Transparence & Pédagogie")
        st.info("NovelForge est un outil d'assistance à l'édition développé dans le cadre d'un projet de Deep Learning.")
        st.markdown(
            """
            **Comment ça marche ?**
            L'IA a été entraînée sur des dizaines de milliers de synopsis issus des bases de données de Light Novels et de Manhwas. 
            Elle a appris à associer le vocabulaire et le contexte d'un texte à ses genres littéraires (Action, Romance, Isekai, etc.).

            **Limites de l'IA (Transparence)**
            - **Biais d'entraînement :** L'IA est performante sur les tropes asiatiques (Manhwa/Light Novel) en anglais. Elle sera moins précise sur de la littérature classique française.
            - **L'ordre des mots :** Les modèles 'Rapides' (TF-IDF) ignorent l'ordre des mots et cherchent des mots-clés. Les modèles 'Avancés' (Transformers) comprennent le contexte.
            - **Usage recommandé :** Les probabilités affichées sont des suggestions d'aide au référencement. Une validation humaine reste indispensable.
            """
        )

if __name__ == "__main__":
    main()
