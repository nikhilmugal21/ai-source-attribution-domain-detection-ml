"""Streamlit deployment application for AI-text detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Use CPU inference on Streamlit Community Cloud.
os.environ.setdefault("FORCE_TENSORFLOW_CPU", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st

from src.inference.predict import CombinedTextPredictor


# ---------------------------------------------------------
# Application paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Text Detection",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_predictor() -> CombinedTextPredictor:
    """Load all trained models once and reuse them."""
    return CombinedTextPredictor(model_dir=MODEL_DIR)


def percentage(value: Any) -> str:
    """Convert a probability into a readable percentage."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Not available"


def display_probabilities(probabilities: dict[str, float]) -> None:
    """Display class probabilities using progress bars."""
    if not probabilities:
        st.caption("Probability values are not available.")
        return

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for label, probability in sorted_probabilities:
        safe_probability = min(max(float(probability), 0.0), 1.0)

        st.write(
            f"**{str(label).replace('-', ' ').title()}** — "
            f"{percentage(safe_probability)}"
        )
        st.progress(safe_probability)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🧠 AI-Generated Text Detection System")

st.write(
    "Analyse whether a passage is human-authored or AI-generated. "
    "For AI predictions, the application also estimates the likely source "
    "and writing domain."
)

st.info(
    "For more reliable results, enter at least 50 words. "
    "Predictions are probabilistic and should not be treated as definitive "
    "proof of authorship."
)


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
try:
    predictor = load_predictor()
except Exception as error:
    st.error("The trained models could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Text input
# ---------------------------------------------------------
text = st.text_area(
    "Enter text for analysis",
    height=260,
    placeholder=(
        "Paste an academic paragraph, news passage, social-media post, "
        "or another piece of writing here..."
    ),
)

word_count = len(text.split()) if text.strip() else 0
character_count = len(text)

count_column_1, count_column_2 = st.columns(2)

with count_column_1:
    st.metric("Word count", word_count)

with count_column_2:
    st.metric("Character count", character_count)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
analyse_button = st.button(
    "Analyse Text",
    type="primary",
    use_container_width=True,
)

if analyse_button:
    cleaned_text = text.strip()

    if not cleaned_text:
        st.error("Please enter some text before starting the analysis.")
        st.stop()

    if word_count < 30:
        st.error(
            "Please enter at least 30 words. Very short text does not provide "
            "enough information for reliable neural and stylometric analysis."
        )
        st.stop()

    if word_count < 50:
        st.warning(
            "This passage contains fewer than 50 words. "
            "The result may be less reliable."
        )

    try:
        with st.spinner("Analysing the text..."):
            result = predictor.predict(cleaned_text)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        st.stop()

    st.success("Analysis completed successfully.")

    # -----------------------------------------------------
    # Main prediction
    # -----------------------------------------------------
    prediction = str(result["prediction"]).upper()
    confidence = float(result["confidence"])

    st.subheader("Main Prediction")

    main_column_1, main_column_2, main_column_3 = st.columns(3)

    with main_column_1:
        st.metric("Authorship", prediction)

    with main_column_2:
        st.metric("Confidence", percentage(confidence))

    with main_column_3:
        source_value = result.get("source")

        if prediction == "AI" and source_value:
            displayed_source = str(source_value).replace("-", " ").title()
        else:
            displayed_source = "Not applicable"

        st.metric("Likely source", displayed_source)

    st.progress(min(max(confidence, 0.0), 1.0))

    # -----------------------------------------------------
    # Detailed model results
    # -----------------------------------------------------
    binary_result = result["binary_detection"]
    source_result = result["source_attribution"]
    domain_result = result["domain_classification"]
    stylometric_result = result["stylometric_detection"]

    tab_binary, tab_source, tab_domain, tab_style = st.tabs(
        [
            "Neural Detection",
            "Source Attribution",
            "Domain Classification",
            "Stylometric Analysis",
        ]
    )

    with tab_binary:
        st.subheader("GRU Human-vs-AI Detection")

        st.write(
            f"Predicted class: **{binary_result['label'].upper()}**"
        )
        st.write(
            f"Confidence: **{percentage(binary_result['confidence'])}**"
        )

        display_probabilities(binary_result.get("probabilities", {}))

    with tab_source:
        st.subheader("AI Source Attribution")

        if source_result.get("applied"):
            source_label = str(source_result["label"]).replace("-", " ").title()

            st.write(f"Likely source: **{source_label}**")
            st.write(
                f"Confidence: **{percentage(source_result['confidence'])}**"
            )

            display_probabilities(source_result.get("probabilities", {}))
        else:
            st.info(
                source_result.get(
                    "reason",
                    "Source attribution was not applied.",
                )
            )

    with tab_domain:
        st.subheader("Writing Domain")

        if domain_result.get("applied"):
            domain_label = str(domain_result["label"]).title()

            st.write(f"Predicted domain: **{domain_label}**")
            st.write(
                f"Confidence: **{percentage(domain_result['confidence'])}**"
            )

            display_probabilities(domain_result.get("probabilities", {}))
        else:
            st.info("The domain classifier is not available.")

    with tab_style:
        st.subheader("Independent Stylometric Detection")

        if stylometric_result.get("applied"):
            style_label = str(stylometric_result["label"]).upper()

            st.write(f"Stylometric prediction: **{style_label}**")
            st.write(
                f"Confidence: "
                f"**{percentage(stylometric_result['confidence'])}**"
            )

            display_probabilities(
                stylometric_result.get("probabilities", {})
            )

            if (
                str(binary_result["label"]).lower()
                != str(stylometric_result["label"]).lower()
            ):
                st.warning(
                    "The GRU and stylometric models disagree. "
                    "The main result uses the GRU prediction, while the "
                    "stylometric result should be considered supporting evidence."
                )
        else:
            st.info("The stylometric classifier is not available.")

    # -----------------------------------------------------
    # Text statistics
    # -----------------------------------------------------
    statistics = result["text_statistics"]

    with st.expander("View text statistics"):
        statistics_column_1, statistics_column_2 = st.columns(2)

        with statistics_column_1:
            st.write(
                f"**Words:** {statistics['word_count']}"
            )
            st.write(
                f"**Sentences:** {statistics['sentence_count']}"
            )
            st.write(
                f"**Average sentence length:** "
                f"{statistics['avg_sentence_length']}"
            )

        with statistics_column_2:
            st.write(
                f"**Average word length:** "
                f"{statistics['avg_word_length']}"
            )
            st.write(
                f"**Lexical diversity:** "
                f"{statistics['lexical_diversity']}"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Developed as part of the MITU internship project on human-vs-AI "
    "text detection using GRU, source attribution, domain classification, "
    "and stylometric machine learning."
)
