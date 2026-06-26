"""Reusable prediction service for the MITU AI-text detection project.

The service loads model artifacts once and exposes a simple ``predict_text``
function that can be called by FastAPI, Streamlit, tests, or a CLI.

Expected project structure (relative to the repository root)::

    models/
      binary_classifier.joblib
      binary_tfidf_vectorizer.joblib          # optional when classifier is a Pipeline
      source_classifier.joblib
      source_tfidf_vectorizer.joblib          # optional when classifier is a Pipeline
      domain_classifier.joblib                # optional
      domain_tfidf_vectorizer.joblib          # optional
      stylometric_classifier.joblib           # optional
      stylometric_scaler.joblib               # optional

Several common alternative filenames are also detected automatically.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    import textstat
except ImportError:  # Optional: manual readability fallback is used.
    textstat = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

logger = logging.getLogger(__name__)


class ModelArtifactError(RuntimeError):
    """Raised when required model artifacts cannot be loaded."""


def _first_existing(model_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Return the first existing artifact from a list of candidate names."""
    for candidate in candidates:
        path = model_dir / candidate
        if path.exists():
            return path
    return None


def _load_optional(model_dir: Path, candidates: Sequence[str]) -> Tuple[Any, Optional[Path]]:
    """Load the first matching joblib/pickle artifact, or return ``(None, None)``."""
    path = _first_existing(model_dir, candidates)
    if path is None:
        return None, None
    try:
        return joblib.load(path), path
    except Exception as exc:  # pragma: no cover - exact backend error varies
        raise ModelArtifactError(f"Could not load model artifact: {path}\n{exc}") from exc


def _normalise_label(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _estimate_syllables(word: str) -> int:
    """Estimate English syllables using the rule used in Notebook 05."""
    cleaned = re.sub(r"[^a-z]", "", str(word).lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    cleaned = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", cleaned)
    cleaned = re.sub(r"^y", "", cleaned)
    groups = re.findall(r"[aeiouy]+", cleaned)
    return max(1, len(groups))


def _manual_readability(
    word_count: int,
    sentence_count: int,
    syllable_count: int,
) -> Tuple[float, float]:
    """Return Flesch Reading Ease and Flesch-Kincaid Grade."""
    if word_count <= 0:
        return 0.0, 0.0

    safe_sentence_count = max(sentence_count, 1)
    words_per_sentence = word_count / safe_sentence_count
    syllables_per_word = syllable_count / word_count

    reading_ease = (
        206.835
        - (1.015 * words_per_sentence)
        - (84.6 * syllables_per_word)
    )
    grade = (
        (0.39 * words_per_sentence)
        + (11.8 * syllables_per_word)
        - 15.59
    )
    return reading_ease, grade


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """Extract the same inference features used by Notebook 05.

    Raw text is used so punctuation, casing, paragraph breaks, contractions,
    and non-ASCII characters remain available to the stylometric model.
    Aliases are included because earlier notebooks used a few different names.
    """
    text = "" if text is None else str(text)

    # Notebook 05 tokenisation rules.
    words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)
    lower_words = [word.lower() for word in words]

    sentence_parts = [
        part.strip()
        for part in re.split(r"[.!?]\s+|\n+", text.strip())
        if part.strip()
    ]
    if not sentence_parts and text.strip():
        sentence_parts = [text.strip()]

    paragraph_parts = [
        part.strip()
        for part in re.split(r"\n\s*\n", text.strip())
        if part.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentence_parts)
    paragraph_count = len(paragraph_parts)
    safe_word_count = max(word_count, 1)
    safe_sentence_count = max(sentence_count, 1)

    char_count = len(text)
    whitespace_count = sum(char.isspace() for char in text)
    char_count_no_spaces = sum(not char.isspace() for char in text)
    safe_char_count = max(char_count, 1)
    safe_non_space_count = max(char_count_no_spaces, 1)

    word_lengths = [len(word) for word in words]
    sentence_lengths = [
        len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", sentence))
        for sentence in sentence_parts
    ]
    if not sentence_lengths and word_count:
        sentence_lengths = [word_count]

    avg_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0
    word_length_std = float(np.std(word_lengths)) if word_lengths else 0.0
    avg_sentence_length = word_count / safe_sentence_count
    sentence_length_std = (
        float(np.std(sentence_lengths)) if sentence_lengths else 0.0
    )
    min_sentence_length = min(sentence_lengths) if sentence_lengths else 0
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0

    frequencies = Counter(lower_words)
    unique_word_count = len(frequencies)
    hapax_count = sum(count == 1 for count in frequencies.values())

    type_token_ratio = unique_word_count / safe_word_count
    root_type_token_ratio = unique_word_count / math.sqrt(safe_word_count)
    hapax_legomena_ratio = hapax_count / safe_word_count
    long_word_ratio = sum(len(word) >= 7 for word in words) / safe_word_count
    short_word_ratio = sum(len(word) <= 3 for word in words) / safe_word_count

    stopword_count = sum(word in ENGLISH_STOP_WORDS for word in lower_words)
    stopword_ratio = stopword_count / safe_word_count
    lexical_density_proxy = (word_count - stopword_count) / safe_word_count

    uppercase_char_count = sum(char.isupper() for char in text)
    uppercase_word_count = sum(
        word.isupper() and any(char.isalpha() for char in word)
        for word in words
    )
    digit_count = sum(char.isdigit() for char in text)
    newline_count = text.count("\n")
    non_ascii_count = sum(ord(char) > 127 for char in text)

    punctuation_count = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
    comma_count = text.count(",")
    period_count = text.count(".")
    question_count = text.count("?")
    exclamation_count = text.count("!")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    hyphen_count = sum(text.count(mark) for mark in ("-", "–", "—"))
    apostrophe_count = text.count("'") + text.count("’")
    quotation_count = sum(text.count(mark) for mark in ('"', "“", "”"))
    parenthesis_count = sum(text.count(mark) for mark in ("(", ")"))
    ellipsis_count = len(re.findall(r"(?:\.{3,}|…)", text))
    repeated_punctuation_count = len(re.findall(r"[!?.,;:]{2,}", text))
    repeated_space_count = len(re.findall(r" {2,}", text))

    per_100 = 100.0 / safe_word_count

    contraction_count = sum(("'" in word or "’" in word) for word in words)
    contraction_ratio = contraction_count / safe_word_count

    first_person = {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
    }
    second_person = {
        "you", "your", "yours", "yourself", "yourselves",
    }
    third_person = {
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    }

    first_person_pronoun_ratio = (
        sum(word in first_person for word in lower_words) / safe_word_count
    )
    second_person_pronoun_ratio = (
        sum(word in second_person for word in lower_words) / safe_word_count
    )
    third_person_pronoun_ratio = (
        sum(word in third_person for word in lower_words) / safe_word_count
    )

    transition_markers = (
        "however", "therefore", "furthermore", "moreover",
        "consequently", "additionally", "nevertheless", "nonetheless",
        "thus", "hence", "meanwhile", "similarly", "likewise",
        "in addition", "for example", "for instance",
        "on the other hand", "as a result", "in conclusion",
        "first", "second", "finally",
    )
    lower_text = text.lower()
    transition_count = 0
    for marker in transition_markers:
        pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"
        transition_count += len(re.findall(pattern, lower_text))
    transition_word_ratio = transition_count / safe_word_count

    capitalised_sentence_count = 0
    for sentence in sentence_parts:
        first_alpha = next((char for char in sentence if char.isalpha()), "")
        if first_alpha and first_alpha.isupper():
            capitalised_sentence_count += 1
    sentence_initial_capital_ratio = (
        capitalised_sentence_count / safe_sentence_count
    )

    syllable_count = sum(_estimate_syllables(word) for word in words)
    syllables_per_word = syllable_count / safe_word_count
    manual_reading_ease, manual_grade = _manual_readability(
        word_count=word_count,
        sentence_count=sentence_count,
        syllable_count=syllable_count,
    )

    if textstat is not None and text.strip():
        try:
            flesch_reading_ease = float(textstat.flesch_reading_ease(text))
            flesch_kincaid_grade = float(textstat.flesch_kincaid_grade(text))
        except Exception:
            flesch_reading_ease = manual_reading_ease
            flesch_kincaid_grade = manual_grade
    else:
        flesch_reading_ease = manual_reading_ease
        flesch_kincaid_grade = manual_grade

    features = {
        # Basic counts and aliases
        "char_count": char_count,
        "character_count": char_count,
        "text_length": char_count,
        "char_count_no_spaces": char_count_no_spaces,
        "non_space_char_count": char_count_no_spaces,
        "whitespace_count": whitespace_count,
        "whitespace_ratio": whitespace_count / safe_char_count,
        "word_count": word_count,
        "word_count_stylometric": word_count,
        "num_words": word_count,
        "sentence_count": sentence_count,
        "sentence_count_stylometric": sentence_count,
        "num_sentences": sentence_count,
        "paragraph_count": paragraph_count,
        "unique_word_count": unique_word_count,

        # Word and sentence length
        "avg_word_length": avg_word_length,
        "average_word_length": avg_word_length,
        "word_length_std": word_length_std,
        "avg_sentence_length": avg_sentence_length,
        "average_sentence_length": avg_sentence_length,
        "avg_sentence_length_stylometric": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "sentence_length_standard_deviation": sentence_length_std,
        "min_sentence_length": min_sentence_length,
        "max_sentence_length": max_sentence_length,

        # Vocabulary and lexical composition
        "lexical_diversity": type_token_ratio,
        "type_token_ratio": type_token_ratio,
        "ttr": type_token_ratio,
        "root_type_token_ratio": root_type_token_ratio,
        "hapax_legomena_ratio": hapax_legomena_ratio,
        "hapax_ratio": hapax_legomena_ratio,
        "long_word_ratio": long_word_ratio,
        "short_word_ratio": short_word_ratio,
        "stopword_ratio": stopword_ratio,
        "lexical_density_proxy": lexical_density_proxy,

        # Casing and character composition
        "uppercase_count": uppercase_char_count,
        "uppercase_ratio": uppercase_char_count / safe_non_space_count,
        "uppercase_char_ratio": uppercase_char_count / safe_non_space_count,
        "uppercase_word_ratio": uppercase_word_count / safe_word_count,
        "digit_count": digit_count,
        "digit_ratio": digit_count / safe_non_space_count,
        "digit_char_ratio": digit_count / safe_non_space_count,
        "newline_count": newline_count,
        "newline_ratio": newline_count / safe_char_count,
        "non_ascii_char_ratio": non_ascii_count / safe_char_count,

        # Punctuation
        "punctuation_count": punctuation_count,
        "punctuation_ratio": punctuation_count / safe_char_count,
        "punctuation_char_ratio": punctuation_count / safe_char_count,
        "comma_count": comma_count,
        "period_count": period_count,
        "question_count": question_count,
        "question_mark_count": question_count,
        "exclamation_count": exclamation_count,
        "exclamation_mark_count": exclamation_count,
        "semicolon_count": semicolon_count,
        "colon_count": colon_count,
        "quote_count": quotation_count,
        "repeated_punctuation_count": repeated_punctuation_count,
        "repeated_space_count": repeated_space_count,
        "comma_per_100_words": comma_count * per_100,
        "period_per_100_words": period_count * per_100,
        "question_per_100_words": question_count * per_100,
        "exclamation_per_100_words": exclamation_count * per_100,
        "semicolon_per_100_words": semicolon_count * per_100,
        "colon_per_100_words": colon_count * per_100,
        "hyphen_per_100_words": hyphen_count * per_100,
        "apostrophe_per_100_words": apostrophe_count * per_100,
        "quotation_per_100_words": quotation_count * per_100,
        "parenthesis_per_100_words": parenthesis_count * per_100,
        "ellipsis_per_100_words": ellipsis_count * per_100,
        "repeated_punctuation_per_100_words": (
            repeated_punctuation_count * per_100
        ),

        # Function words and discourse
        "contraction_ratio": contraction_ratio,
        "first_person_pronoun_ratio": first_person_pronoun_ratio,
        "second_person_pronoun_ratio": second_person_pronoun_ratio,
        "third_person_pronoun_ratio": third_person_pronoun_ratio,
        "transition_word_ratio": transition_word_ratio,
        "transition_markers_per_100_words": transition_count * per_100,
        "sentence_initial_capital_ratio": sentence_initial_capital_ratio,

        # Readability
        "syllables_per_word": syllables_per_word,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
    }

    return {name: _safe_float(value) for name, value in features.items()}


def _expected_feature_names(model: Any, scaler: Any = None) -> Optional[List[str]]:
    """Read saved training feature names when scikit-learn exposes them."""
    for obj in (scaler, model):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
    return None


def _prepare_stylometric_input(text: str, model: Any, scaler: Any = None) -> Any:
    feature_map = extract_stylometric_features(text)
    expected_names = _expected_feature_names(model=model, scaler=scaler)

    if expected_names:
        missing = [name for name in expected_names if name not in feature_map]
        if missing:
            raise ModelArtifactError(
                "The stylometric model expects features that are not available in "
                f"src/inference/predict.py: {missing}. Update "
                "extract_stylometric_features() to match Notebook 05 exactly."
            )
        frame = pd.DataFrame([[feature_map[name] for name in expected_names]], columns=expected_names)
    else:
        # Stable default order for models saved without feature_names_in_.
        default_names = [
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "avg_word_length",
            "lexical_diversity",
            "punctuation_ratio",
            "uppercase_ratio",
            "digit_ratio",
        ]
        frame = pd.DataFrame([[feature_map[name] for name in default_names]], columns=default_names)

    return scaler.transform(frame) if scaler is not None else frame


def _is_text_pipeline(model: Any) -> bool:
    """Return True when an artifact appears to accept raw text directly."""
    steps = getattr(model, "steps", None)
    if steps:
        step_names = {str(name).lower() for name, _ in steps}
        return any("tfidf" in name or "vector" in name or "count" in name for name in step_names)
    return False


def _prepare_text_input(text: str, classifier: Any, vectorizer: Any = None) -> Any:
    if vectorizer is not None:
        return vectorizer.transform([text])
    if _is_text_pipeline(classifier):
        return [text]
    # Some notebook exports save a Pipeline-like wrapper without ``steps``.
    return [text]


def _class_probabilities(model: Any, model_input: Any) -> Dict[str, float]:
    classes = list(getattr(model, "classes_", []))

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(model_input))[0]
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(model_input))
        if decision.ndim == 1:
            if decision.size == 1:
                positive = 1.0 / (1.0 + np.exp(-decision[0]))
                raw = np.array([1.0 - positive, positive])
            else:
                raw = _softmax(decision)
        else:
            raw = _softmax(decision)[0]
    else:
        prediction = model.predict(model_input)[0]
        return {_normalise_label(prediction): 1.0}

    if not classes:
        classes = list(range(len(raw)))

    return {
        _normalise_label(label): round(_safe_float(probability), 6)
        for label, probability in zip(classes, raw)
    }


def _predicted_label(model: Any, model_input: Any) -> str:
    prediction = model.predict(model_input)[0]
    return _normalise_label(prediction)


def _canonical_binary_label(label: str) -> str:
    normalised = _normalise_label(label)
    if normalised in {"1", "ai", "ai-generated", "generated", "machine"}:
        return "ai"
    if normalised in {"0", "human", "human-authored", "authentic"}:
        return "human"
    return normalised


def _canonical_source_label(label: str) -> str:
    normalised = _normalise_label(label)
    aliases = {
        "gemini": "gemini-2.0",
        "gemini2": "gemini-2.0",
        "gemini-2": "gemini-2.0",
        "gpt4o": "gpt-4o",
        "gpt-4-o": "gpt-4o",
        "gpt55": "gpt-5.5-thinking",
        "gpt-5.5": "gpt-5.5-thinking",
        "gpt-5-5-thinking": "gpt-5.5-thinking",
    }
    return aliases.get(normalised, normalised)


def _probability_for_binary_label(probabilities: Mapping[str, float], target: str) -> float:
    for label, probability in probabilities.items():
        if _canonical_binary_label(label) == target:
            return _safe_float(probability)
    return 0.0


def _decode_label(label: Any, encoder: Any = None) -> str:
    """Decode a numeric class with a saved LabelEncoder when available."""
    if encoder is None:
        return _normalise_label(label)
    try:
        decoded = encoder.inverse_transform([label])[0]
        return _normalise_label(decoded)
    except Exception:
        try:
            decoded = encoder.inverse_transform([int(label)])[0]
            return _normalise_label(decoded)
        except Exception:
            return _normalise_label(label)


def _decoded_probabilities(
    probabilities: Mapping[str, float], encoder: Any = None, source: bool = False
) -> Dict[str, float]:
    decoded: Dict[str, float] = {}
    for raw_label, probability in probabilities.items():
        label = _decode_label(raw_label, encoder)
        label = _canonical_source_label(label) if source else label
        decoded[label] = round(decoded.get(label, 0.0) + _safe_float(probability), 6)
    return decoded


class KerasBinaryTextClassifier:
    """Small adapter that makes a saved Keras text model look like sklearn."""

    classes_ = np.asarray(["human", "ai"], dtype=object)

    def __init__(self, model_path: Path, tokenizer: Any, max_length: Optional[int] = None):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ModelArtifactError(
                "A .keras binary model was found, but TensorFlow is not installed. "
                "Install it with: python -m pip install tensorflow"
            ) from exc

        # On macOS, TensorFlow/Metal initialization can occasionally stall when
        # Uvicorn reloads the process. CPU inference is more predictable for this
        # small deployment model. Set FORCE_TENSORFLOW_CPU=0 to allow GPU use.
        if os.getenv("FORCE_TENSORFLOW_CPU", "1") == "1":
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError:
                # TensorFlow may already be initialized; continuing is safe.
                pass

        self.model_path = Path(model_path)
        logger.info("Loading Keras model from %s", self.model_path)
        self.model = load_model(self.model_path, compile=False)
        self.tokenizer = tokenizer
        self._prediction_lock = Lock()

        inferred_length = None
        try:
            input_shape = self.model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
                inferred_length = int(input_shape[1])
        except Exception:
            inferred_length = None

        self.max_length = int(max_length or inferred_length or 250)

    def _prepare(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError as exc:
            raise ModelArtifactError("TensorFlow is required for Keras inference.") from exc

        sequences = self.tokenizer.texts_to_sequences([str(text) for text in texts])
        return pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        model_input = self._prepare(texts)

        # Calling the model directly avoids Keras' internal prediction dataset
        # setup, which is slower and can hang in some macOS/Uvicorn combinations.
        with self._prediction_lock:
            output = self.model(model_input, training=False)

        if hasattr(output, "numpy"):
            output = output.numpy()
        raw = np.asarray(output, dtype=float)

        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        if raw.shape[1] == 1:
            ai_probability = raw[:, 0]
            if np.any((ai_probability < 0.0) | (ai_probability > 1.0)):
                ai_probability = 1.0 / (1.0 + np.exp(-ai_probability))
            ai_probability = np.clip(ai_probability, 0.0, 1.0)
            return np.column_stack([1.0 - ai_probability, ai_probability])

        if raw.shape[1] == 2:
            row_sums = raw.sum(axis=1, keepdims=True)
            looks_like_probabilities = (
                np.all(raw >= 0.0)
                and np.all(raw <= 1.0)
                and np.allclose(row_sums, 1.0, atol=1e-3)
            )
            return raw if looks_like_probabilities else _softmax(raw)

        raise ModelArtifactError(
            f"Expected one or two binary output units, found shape {raw.shape}."
        )

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]


def _load_pickle_or_joblib(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as file:
            return pickle.load(file)


def _read_max_length(model_dir: Path, model: Any = None) -> Optional[int]:
    candidates = (
        "sequence_config.json",
        "inference_config.json",
        "model_metadata.json",
        "training_metadata.json",
    )
    keys = ("max_length", "max_sequence_length", "sequence_length", "MAX_LENGTH")
    for filename in candidates:
        path = model_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            for key in keys:
                if key in payload and payload[key] is not None:
                    return int(payload[key])
        except Exception:
            continue
    return None


def _load_keras_binary_fallback(model_dir: Path) -> Tuple[Any, Optional[Path], Optional[Path]]:
    model_path = _first_existing(
        model_dir,
        (
            "binary_model.keras",
            "gru_binary_model.keras",
            "gru_model.keras",
            "lstm_binary_model.keras",
            "lstm_model.keras",
            "best_gru_model.keras",
            "best_lstm_model.keras",
        ),
    )
    if model_path is None:
        return None, None, None

    tokenizer_path = _first_existing(
        model_dir,
        (
            "binary_tokenizer.joblib",
            "text_tokenizer.joblib",
            "tokenizer.joblib",
            "binary_tokenizer.pkl",
            "text_tokenizer.pkl",
            "tokenizer.pkl",
            "tokenizer.pickle",
        ),
    )
    if tokenizer_path is None:
        raise ModelArtifactError(
            f"Found Keras model {model_path.name}, but no saved tokenizer was found "
            f"in {model_dir}. Save the fitted training tokenizer as tokenizer.joblib."
        )

    tokenizer = _load_pickle_or_joblib(tokenizer_path)
    max_length = _read_max_length(model_dir)
    classifier = KerasBinaryTextClassifier(
        model_path=model_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return classifier, model_path, tokenizer_path


@dataclass
class LoadedArtifacts:
    binary_classifier: Any
    binary_vectorizer: Any = None
    source_classifier: Any = None
    source_vectorizer: Any = None
    source_label_encoder: Any = None
    domain_classifier: Any = None
    domain_vectorizer: Any = None
    domain_label_encoder: Any = None
    stylometric_classifier: Any = None
    stylometric_scaler: Any = None
    paths: Optional[Dict[str, str]] = None


class CombinedTextPredictor:
    """Load all project models once and make combined predictions."""

    BINARY_CLASSIFIER_NAMES = (
        "binary_classifier.joblib",
        "binary_model.joblib",
        "tfidf_binary_classifier.joblib",
        "binary_pipeline.joblib",
        "binary_text_pipeline.joblib",
    )
    BINARY_VECTORIZER_NAMES = (
        "binary_tfidf_vectorizer.joblib",
        "binary_vectorizer.joblib",
        "tfidf_binary_vectorizer.joblib",
    )
    SOURCE_CLASSIFIER_NAMES = (
        "source_classifier.joblib",
        "source_model.joblib",
        "source_attribution_classifier.joblib",
        "source_pipeline.joblib",
        "source_attribution_pipeline.joblib",
    )
    SOURCE_VECTORIZER_NAMES = (
        "source_tfidf_vectorizer.joblib",
        "source_vectorizer.joblib",
        "source_attribution_vectorizer.joblib",
    )
    SOURCE_LABEL_ENCODER_NAMES = (
        "source_label_encoder.joblib",
        "source_encoder.joblib",
        "source_attribution_label_encoder.joblib",
    )
    DOMAIN_CLASSIFIER_NAMES = (
        "domain_classifier.joblib",
        "domain_model.joblib",
        "domain_pipeline.joblib",
    )
    DOMAIN_VECTORIZER_NAMES = (
        "domain_tfidf_vectorizer.joblib",
        "domain_vectorizer.joblib",
    )
    DOMAIN_LABEL_ENCODER_NAMES = (
        "domain_label_encoder.joblib",
        "domain_encoder.joblib",
    )
    STYLOMETRIC_CLASSIFIER_NAMES = (
        "stylometric_classifier.joblib",
        "stylometry_classifier.joblib",
        "stylometric_model.joblib",
        "stylometry_model.joblib",
        "stylometric_pipeline.joblib",
    )
    STYLOMETRIC_SCALER_NAMES = (
        "stylometric_scaler.joblib",
        "stylometry_scaler.joblib",
        "feature_scaler.joblib",
    )

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> LoadedArtifacts:
        if not self.model_dir.exists():
            raise ModelArtifactError(
                f"Model directory does not exist: {self.model_dir}\n"
                "Copy the exported notebook artifacts into the project's models/ folder."
            )

        binary_classifier, binary_classifier_path = _load_optional(
            self.model_dir, self.BINARY_CLASSIFIER_NAMES
        )
        binary_vectorizer, binary_vectorizer_path = _load_optional(
            self.model_dir, self.BINARY_VECTORIZER_NAMES
        )

        # When no sklearn/joblib binary classifier exists, load a saved LSTM/GRU.
        keras_tokenizer_path = None
        if binary_classifier is None:
            binary_classifier, binary_classifier_path, keras_tokenizer_path = (
                _load_keras_binary_fallback(self.model_dir)
            )

        source_classifier, source_classifier_path = _load_optional(
            self.model_dir, self.SOURCE_CLASSIFIER_NAMES
        )
        source_vectorizer, source_vectorizer_path = _load_optional(
            self.model_dir, self.SOURCE_VECTORIZER_NAMES
        )
        source_label_encoder, source_label_encoder_path = _load_optional(
            self.model_dir, self.SOURCE_LABEL_ENCODER_NAMES
        )
        domain_classifier, domain_classifier_path = _load_optional(
            self.model_dir, self.DOMAIN_CLASSIFIER_NAMES
        )
        domain_vectorizer, domain_vectorizer_path = _load_optional(
            self.model_dir, self.DOMAIN_VECTORIZER_NAMES
        )
        domain_label_encoder, domain_label_encoder_path = _load_optional(
            self.model_dir, self.DOMAIN_LABEL_ENCODER_NAMES
        )
        stylometric_classifier, stylometric_classifier_path = _load_optional(
            self.model_dir, self.STYLOMETRIC_CLASSIFIER_NAMES
        )
        stylometric_scaler, stylometric_scaler_path = _load_optional(
            self.model_dir, self.STYLOMETRIC_SCALER_NAMES
        )

        if binary_classifier is None:
            expected = ", ".join(self.BINARY_CLASSIFIER_NAMES)
            raise ModelArtifactError(
                "No binary classifier was found.\n"
                f"Searched in: {self.model_dir}\n"
                f"Accepted joblib filenames: {expected}\n"
                "Accepted Keras filenames include gru_model.keras and lstm_model.keras."
            )

        loaded_paths = {
            key: str(path)
            for key, path in {
                "binary_classifier": binary_classifier_path,
                "binary_vectorizer": binary_vectorizer_path,
                "keras_tokenizer": keras_tokenizer_path,
                "source_classifier": source_classifier_path,
                "source_vectorizer": source_vectorizer_path,
                "source_label_encoder": source_label_encoder_path,
                "domain_classifier": domain_classifier_path,
                "domain_vectorizer": domain_vectorizer_path,
                "domain_label_encoder": domain_label_encoder_path,
                "stylometric_classifier": stylometric_classifier_path,
                "stylometric_scaler": stylometric_scaler_path,
            }.items()
            if path is not None
        }

        return LoadedArtifacts(
            binary_classifier=binary_classifier,
            binary_vectorizer=binary_vectorizer,
            source_classifier=source_classifier,
            source_vectorizer=source_vectorizer,
            source_label_encoder=source_label_encoder,
            domain_classifier=domain_classifier,
            domain_vectorizer=domain_vectorizer,
            domain_label_encoder=domain_label_encoder,
            stylometric_classifier=stylometric_classifier,
            stylometric_scaler=stylometric_scaler,
            paths=loaded_paths,
        )

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "model_directory": str(self.model_dir),
            "loaded_artifacts": self.artifacts.paths or {},
            "available_tasks": {
                "binary_detection": self.artifacts.binary_classifier is not None,
                "source_attribution": self.artifacts.source_classifier is not None,
                "domain_classification": self.artifacts.domain_classifier is not None,
                "stylometric_detection": self.artifacts.stylometric_classifier is not None,
            },
        }

    def _predict_binary(self, text: str) -> Dict[str, Any]:
        model_input = _prepare_text_input(
            text,
            classifier=self.artifacts.binary_classifier,
            vectorizer=self.artifacts.binary_vectorizer,
        )

        classifier = self.artifacts.binary_classifier
        if isinstance(classifier, KerasBinaryTextClassifier):
            # Run Keras only once. The earlier version called the neural model
            # twice for every request: once for the label and again for probabilities.
            probability_row = np.asarray(classifier.predict_proba(model_input))[0]
            class_index = int(np.argmax(probability_row))
            raw_label = _normalise_label(classifier.classes_[class_index])
            probabilities = {
                _normalise_label(label): round(_safe_float(probability), 6)
                for label, probability in zip(classifier.classes_, probability_row)
            }
        else:
            raw_label = _predicted_label(classifier, model_input)
            probabilities = _class_probabilities(classifier, model_input)

        label = _canonical_binary_label(raw_label)
        confidence = max(probabilities.values()) if probabilities else 1.0
        ai_probability = _probability_for_binary_label(probabilities, "ai")
        human_probability = _probability_for_binary_label(probabilities, "human")

        return {
            "label": label,
            "confidence": round(confidence, 6),
            "ai_probability": round(ai_probability, 6),
            "human_probability": round(human_probability, 6),
            "probabilities": {
                _canonical_binary_label(key): value for key, value in probabilities.items()
            },
        }

    def _predict_source(self, text: str, binary_label: str) -> Dict[str, Any]:
        if binary_label == "human":
            return {
                "label": "human",
                "confidence": 1.0,
                "probabilities": {"human": 1.0},
                "applied": False,
                "reason": "Source attribution is only required after an AI prediction.",
            }

        if self.artifacts.source_classifier is None:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "applied": False,
                "reason": "Source-attribution model is not installed.",
            }

        model_input = _prepare_text_input(
            text,
            classifier=self.artifacts.source_classifier,
            vectorizer=self.artifacts.source_vectorizer,
        )
        raw_label = self.artifacts.source_classifier.predict(model_input)[0]
        probabilities = _class_probabilities(self.artifacts.source_classifier, model_input)
        canonical_probabilities = _decoded_probabilities(
            probabilities,
            encoder=self.artifacts.source_label_encoder,
            source=True,
        )
        label = _canonical_source_label(
            _decode_label(raw_label, self.artifacts.source_label_encoder)
        )

        return {
            "label": label,
            "confidence": round(max(canonical_probabilities.values()), 6)
            if canonical_probabilities
            else 1.0,
            "probabilities": canonical_probabilities,
            "applied": True,
            "reason": None,
        }

    def _predict_domain(self, text: str) -> Dict[str, Any]:
        if self.artifacts.domain_classifier is None:
            return {
                "label": None,
                "confidence": None,
                "probabilities": {},
                "applied": False,
            }

        model_input = _prepare_text_input(
            text,
            classifier=self.artifacts.domain_classifier,
            vectorizer=self.artifacts.domain_vectorizer,
        )
        raw_label = self.artifacts.domain_classifier.predict(model_input)[0]
        label = _decode_label(raw_label, self.artifacts.domain_label_encoder)
        probabilities = _decoded_probabilities(
            _class_probabilities(self.artifacts.domain_classifier, model_input),
            encoder=self.artifacts.domain_label_encoder,
        )
        return {
            "label": label,
            "confidence": round(max(probabilities.values()), 6) if probabilities else 1.0,
            "probabilities": probabilities,
            "applied": True,
        }

    def _predict_stylometric(self, text: str) -> Dict[str, Any]:
        if self.artifacts.stylometric_classifier is None:
            return {
                "label": None,
                "confidence": None,
                "ai_probability": None,
                "human_probability": None,
                "probabilities": {},
                "applied": False,
            }

        model_input = _prepare_stylometric_input(
            text,
            model=self.artifacts.stylometric_classifier,
            scaler=self.artifacts.stylometric_scaler,
        )
        raw_label = _predicted_label(self.artifacts.stylometric_classifier, model_input)
        probabilities = _class_probabilities(self.artifacts.stylometric_classifier, model_input)
        label = _canonical_binary_label(raw_label)
        return {
            "label": label,
            "confidence": round(max(probabilities.values()), 6) if probabilities else 1.0,
            "ai_probability": round(_probability_for_binary_label(probabilities, "ai"), 6),
            "human_probability": round(_probability_for_binary_label(probabilities, "human"), 6),
            "probabilities": {
                _canonical_binary_label(key): value for key, value in probabilities.items()
            },
            "applied": True,
        }

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned_text = str(text).strip()
        if len(cleaned_text) < 20:
            raise ValueError("Please provide at least 20 characters of text.")

        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        logger.info("Prediction stage started: binary")
        binary = self._predict_binary(cleaned_text)
        logger.info("Prediction stage completed: binary in %.3fs", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        logger.info("Prediction stage started: source")
        source = self._predict_source(cleaned_text, binary_label=binary["label"])
        logger.info("Prediction stage completed: source in %.3fs", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        logger.info("Prediction stage started: domain")
        domain = self._predict_domain(cleaned_text)
        logger.info("Prediction stage completed: domain in %.3fs", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        logger.info("Prediction stage started: stylometric")
        stylometric = self._predict_stylometric(cleaned_text)
        logger.info("Prediction stage completed: stylometric in %.3fs", time.perf_counter() - stage_start)

        text_features = extract_stylometric_features(cleaned_text)
        logger.info("Prediction completed in %.3fs", time.perf_counter() - total_start)

        return {
            "prediction": binary["label"],
            "confidence": binary["confidence"],
            "source": source["label"],
            "domain": domain["label"],
            "binary_detection": binary,
            "source_attribution": source,
            "domain_classification": domain,
            "stylometric_detection": stylometric,
            "text_statistics": {
                "word_count": int(text_features["word_count"]),
                "sentence_count": int(text_features["sentence_count"]),
                "avg_sentence_length": round(text_features["avg_sentence_length"], 3),
                "avg_word_length": round(text_features["avg_word_length"], 3),
                "lexical_diversity": round(text_features["lexical_diversity"], 4),
            },
        }

    def predict_batch(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.predict(text) for text in texts]


_predictor: Optional[CombinedTextPredictor] = None
_predictor_lock = Lock()


def get_predictor(model_dir: Path | str = DEFAULT_MODEL_DIR) -> CombinedTextPredictor:
    """Return a process-wide predictor so models are not loaded per request."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = CombinedTextPredictor(model_dir=model_dir)
    return _predictor


def predict_text(text: str) -> Dict[str, Any]:
    """Convenience function for Streamlit or command-line use."""
    return get_predictor().predict(text)
