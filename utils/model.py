"""
Model inference engine for ticket classification.
"""
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocessing import TextPreprocessor


class TicketClassifier:
    """Loads model artifacts and performs predictions."""

    def __init__(self, model_path, tokenizer_path, label_encoder_path, params_path):
        self.model = load_model(model_path)
        
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        self.max_sequence_length = params["max_sequence_length"]
        self.preprocessor = TextPreprocessor(params_path)
        self.classes = list(self.label_encoder.classes_)

    def predict(self, text: str) -> dict:
        """
        Classify a support ticket.
        
        Returns:
            dict with department, confidence, and all probabilities
        """
        processed = self.preprocessor.preprocess(text)
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(
            sequence,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        prediction = self.model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        department = self.classes[predicted_class]

        probabilities = {
            dept: float(prob)
            for dept, prob in zip(self.classes, prediction[0])
        }

        return {
            "department": department,
            "confidence": confidence,
            "probabilities": probabilities,
            "processed_text": processed,
        }

    def predict_batch(self, texts: list) -> list:
        """Classify multiple tickets at once."""
        return [self.predict(text) for text in texts]
