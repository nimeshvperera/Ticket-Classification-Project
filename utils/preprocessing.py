"""
Text preprocessing pipeline for support ticket classification.
Mirrors the exact preprocessing from the training notebook.
"""
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Handles all text preprocessing for ticket classification."""

    def __init__(self, params_path=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Keep important words for support ticket context
        important_words = {
            "not", "no", "nor", "very", "urgent", "immediately",
            "cannot", "need", "help", "issue", "problem",
        }
        self.stop_words = self.stop_words - important_words

        # Load saved preprocessing params if provided
        if params_path:
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            self.max_sequence_length = params["max_sequence_length"]
            self.max_words = params["max_words"]
            if "stop_words" in params:
                self.stop_words = set(params["stop_words"])

    def clean_text(self, text: str) -> str:
        """Clean raw ticket text."""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"#\d+", "", text)
        text = re.sub(r"[^a-zA-Z\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize, remove stopwords, and lemmatize."""
        tokens = word_tokenize(text)
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(processed_tokens)

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed
