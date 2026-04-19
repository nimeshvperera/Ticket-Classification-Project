"""
Application configuration settings.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_ticket_model.keras"
TOKENIZER_PATH = MODELS_DIR / "ticket_tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "ticket_label_encoder.pkl"
PREPROCESSING_PARAMS_PATH = MODELS_DIR / "ticket_preprocessing_params.pkl"

# Data directory
DATA_DIR = BASE_DIR / "data"

# Application settings
APP_TITLE = "TicketFlow AI"
APP_SUBTITLE = "Intelligent Support Ticket Classification"
APP_ICON = "🎫"

# Department metadata
DEPARTMENT_CONFIG = {
    "Technical Support": {
        "icon": "🔧",
        "color": "#E63946",
        "bg_color": "#FFF1F2",
        "description": "Hardware, software, and infrastructure issues",
        "priority_keywords": ["crash", "down", "error", "bug", "broken", "urgent"],
    },
    "Sales": {
        "icon": "💰",
        "color": "#2D6A4F",
        "bg_color": "#ECFDF5",
        "description": "Pricing, plans, and purchase inquiries",
        "priority_keywords": ["pricing", "plan", "purchase", "discount", "enterprise"],
    },
    "Billing": {
        "icon": "📄",
        "color": "#E9C46A",
        "bg_color": "#FEFCE8",
        "description": "Payments, invoices, and account charges",
        "priority_keywords": ["charge", "refund", "invoice", "payment", "overcharged"],
    },
    "Customer Service": {
        "icon": "💬",
        "color": "#457B9D",
        "bg_color": "#EFF6FF",
        "description": "General inquiries and feedback",
        "priority_keywords": ["thank", "feedback", "question", "information", "help"],
    },
}
