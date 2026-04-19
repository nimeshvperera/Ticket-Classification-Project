"""
TicketFlow AI — Intelligent Support Ticket Classification
Production-grade Streamlit interface for CNN-LSTM ticket routing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config.settings import (
    MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH,
    PREPROCESSING_PARAMS_PATH, DEPARTMENT_CONFIG,
    APP_TITLE, APP_SUBTITLE, APP_ICON, DATA_DIR,
)
from utils.model import TicketClassifier

# ---------------------------------------------------------------------------
# Page Config & Theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0A0A0F;
        --bg-secondary: #12121A;
        --bg-card: #1A1A26;
        --bg-card-hover: #22222E;
        --border-color: #2A2A3A;
        --border-accent: #3A3A4A;
        --text-primary: #E8E8F0;
        --text-secondary: #8888A0;
        --text-muted: #5A5A72;
        --accent-red: #E63946;
        --accent-green: #2D6A4F;
        --accent-gold: #E9C46A;
        --accent-blue: #457B9D;
        --accent-cyan: #48BFE3;
    }

    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Outfit', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary);
    }

    /* Main header */
    .app-header {
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    .app-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        margin: 0;
    }
    .app-title span {
        color: var(--accent-cyan);
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
        font-weight: 300;
    }

    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--border-accent);
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* Result card */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        border-radius: 4px 0 0 4px;
    }
    .result-card.dept-technical::before { background: var(--accent-red); }
    .result-card.dept-sales::before { background: var(--accent-green); }
    .result-card.dept-billing::before { background: var(--accent-gold); }
    .result-card.dept-customer::before { background: var(--accent-blue); }

    .result-department {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.6rem 0;
        gap: 0.75rem;
    }
    .prob-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-secondary);
        min-width: 140px;
        text-align: right;
    }
    .prob-bar-bg {
        flex: 1;
        height: 24px;
        background: var(--bg-secondary);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        display: flex;
        align-items: center;
        padding-left: 8px;
    }
    .prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-primary);
        min-width: 50px;
        text-align: right;
    }

    /* History table */
    .history-row {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: border-color 0.2s;
    }
    .history-row:hover {
        border-color: var(--border-accent);
    }
    .history-dept-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .history-text {
        flex: 1;
        font-size: 0.9rem;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-muted);
        white-space: nowrap;
    }
    .history-time {
        font-size: 0.75rem;
        color: var(--text-muted);
        white-space: nowrap;
    }

    /* Batch table */
    .batch-result-dept {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-online { background: #22c55e; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Text area styling */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 1px var(--accent-cyan) !important;
    }

    /* Button */
    .stButton > button[kind="primary"] {
        background: var(--accent-cyan) !important;
        color: var(--bg-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #5CD1E5 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-muted);
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* File uploader */
    .stFileUploader {
        background: var(--bg-card);
        border: 1px dashed var(--border-accent);
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: var(--border-color) !important;
    }

    /* Section headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load model artifacts once and cache."""
    return TicketClassifier(
        model_path=str(MODEL_PATH),
        tokenizer_path=str(TOKENIZER_PATH),
        label_encoder_path=str(LABEL_ENCODER_PATH),
        params_path=str(PREPROCESSING_PARAMS_PATH),
    )


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


def add_to_history(text, result):
    st.session_state.history.insert(0, {
        "text": text,
        "department": result["department"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "timestamp": datetime.now().isoformat(),
    })
    # Keep last 100
    st.session_state.history = st.session_state.history[:100]


# ---------------------------------------------------------------------------
# Helper: Department Styling
# ---------------------------------------------------------------------------

def get_dept_class(department):
    mapping = {
        "Technical Support": "dept-technical",
        "Sales": "dept-sales",
        "Billing": "dept-billing",
        "Customer Service": "dept-customer",
    }
    return mapping.get(department, "dept-customer")


def get_dept_color(department):
    config = DEPARTMENT_CONFIG.get(department, {})
    return config.get("color", "#48BFE3")


def get_dept_icon(department):
    config = DEPARTMENT_CONFIG.get(department, {})
    return config.get("icon", "📋")


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🎫 Ticket<span>Flow</span> AI</div>
        <div class="app-subtitle">
            <span class="status-dot status-online"></span>
            CNN-LSTM Hybrid Model · 4 Departments · Real-time Classification
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(classifier):
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #E8E8F0;">
                🎫 TicketFlow AI
            </div>
            <div style="font-size: 0.8rem; color: #5A5A72; margin-top: 0.25rem;">
                v1.0.0 · Production
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.5rem;">
            <div class="metric-label">Architecture</div>
            <div style="color: #E8E8F0; font-size: 0.9rem; margin-top: 0.25rem;">CNN-LSTM Hybrid</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.5rem;">
            <div class="metric-label">Departments</div>
            <div style="color: #E8E8F0; font-size: 0.9rem; margin-top: 0.25rem;">{len(classifier.classes)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.5rem;">
            <div class="metric-label">Max Sequence Length</div>
            <div style="color: #E8E8F0; font-size: 0.9rem; margin-top: 0.25rem;">{classifier.max_sequence_length}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-header">Departments</div>', unsafe_allow_html=True)

        for dept, config in DEPARTMENT_CONFIG.items():
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.6rem 0;">
                <span style="font-size: 1.1rem;">{config['icon']}</span>
                <span style="color: {config['color']}; font-size: 0.85rem; font-weight: 500;">{dept}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)

        total = len(st.session_state.history)
        dept_counts = {}
        for h in st.session_state.history:
            dept_counts[h["department"]] = dept_counts.get(h["department"], 0) + 1

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Tickets Classified</div>
        </div>
        """, unsafe_allow_html=True)


def render_probability_bars(probabilities):
    """Render horizontal probability bars for each department."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    max_prob = max(probabilities.values())

    for dept, prob in sorted_probs:
        color = get_dept_color(dept)
        icon = get_dept_icon(dept)
        width = (prob / max_prob) * 100 if max_prob > 0 else 0
        pct = prob * 100

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{icon} {dept}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {width}%; background: {color}20; border-left: 3px solid {color};">
                </div>
            </div>
            <div class="prob-value" style="color: {color};">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def render_result(result, text):
    """Render the classification result card."""
    dept = result["department"]
    conf = result["confidence"]
    dept_class = get_dept_class(dept)
    dept_color = get_dept_color(dept)
    dept_icon = get_dept_icon(dept)
    dept_config = DEPARTMENT_CONFIG.get(dept, {})

    st.markdown(f"""
    <div class="result-card {dept_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                    Routed To
                </div>
                <div class="result-department">
                    {dept_icon} {dept}
                </div>
                <div class="result-confidence">
                    {dept_config.get('description', '')}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: {dept_color};">
                    {conf*100:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_classify(classifier):
    """Single ticket classification page."""
    st.markdown('<div class="section-header">Classify a Ticket</div>', unsafe_allow_html=True)

    ticket_text = st.text_area(
        "Paste or type a support ticket",
        height=140,
        placeholder="e.g., My application keeps crashing when I try to export data. I've tried restarting but the issue persists. Need urgent help!",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_btn = st.button("Classify →", type="primary", use_container_width=True)

    if classify_btn and ticket_text.strip():
        with st.spinner(""):
            result = classifier.predict(ticket_text)
            add_to_history(ticket_text, result)

        render_result(result, ticket_text)

        st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
        render_probability_bars(result["probabilities"])

        # Show preprocessed text in expander
        with st.expander("Preprocessing Details"):
            st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-secondary); 
                        background: var(--bg-secondary); padding: 1rem; border-radius: 6px; border: 1px solid var(--border-color);">
                <span style="color: var(--text-muted);">Original:</span><br>{ticket_text}<br><br>
                <span style="color: var(--text-muted);">Processed:</span><br>{result['processed_text']}
            </div>
            """, unsafe_allow_html=True)

    elif classify_btn:
        st.warning("Please enter a support ticket to classify.")

    # Sample tickets
    st.markdown('<div class="section-header">Try a Sample</div>', unsafe_allow_html=True)

    samples = {
        "🔧 Technical Issue": "My application keeps crashing when I try to export data. I've tried restarting but the error persists. Need urgent technical help!",
        "💰 Sales Inquiry": "I would like to know more about your enterprise pricing plans and volume discounts for our team of 50 people.",
        "📄 Billing Problem": "I was charged twice on my credit card for the same subscription. Please refund the duplicate charge immediately.",
        "💬 General Feedback": "Thank you for the excellent support! My issue was resolved quickly and professionally. Great experience overall.",
    }

    cols = st.columns(len(samples))
    for col, (label, sample_text) in zip(cols, samples.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                with st.spinner(""):
                    result = classifier.predict(sample_text)
                    add_to_history(sample_text, result)

                render_result(result, sample_text)

                st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
                render_probability_bars(result["probabilities"])


def page_batch(classifier):
    """Batch classification page."""
    st.markdown('<div class="section-header">Batch Classification</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">
        Upload a CSV file with a column named <code style="background: var(--bg-card); padding: 2px 6px; border-radius: 4px; color: var(--accent-cyan);">ticket_text</code> 
        to classify multiple tickets at once.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "ticket_text" not in df.columns:
            st.error("CSV must contain a `ticket_text` column.")
            # Show available columns
            st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            return

        st.markdown(f"""
        <div class="metric-card" style="margin: 1rem 0;">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Tickets to Classify</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Classify All →", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            for i, row in df.iterrows():
                text = str(row["ticket_text"])
                result = classifier.predict(text)
                results.append(result)
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.markdown(
                    f'<span style="font-family: JetBrains Mono; font-size: 0.85rem; color: var(--text-muted);">'
                    f'Processing {i+1}/{len(df)}...</span>',
                    unsafe_allow_html=True,
                )

            status_text.markdown(
                '<span style="font-family: JetBrains Mono; font-size: 0.85rem; color: #22c55e;">✓ Complete</span>',
                unsafe_allow_html=True,
            )

            # Build results DataFrame
            df["predicted_department"] = [r["department"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]
            for dept in classifier.classes:
                df[f"prob_{dept}"] = [r["probabilities"][dept] for r in results]

            # Summary metrics
            st.markdown('<div class="section-header">Results Summary</div>', unsafe_allow_html=True)

            dept_counts = df["predicted_department"].value_counts()
            cols = st.columns(len(classifier.classes))
            for col, dept in zip(cols, classifier.classes):
                count = dept_counts.get(dept, 0)
                color = get_dept_color(dept)
                icon = get_dept_icon(dept)
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div class="metric-value" style="color: {color};">{count}</div>
                        <div class="metric-label">{dept}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Avg confidence
            avg_conf = df["confidence"].mean()
            low_conf = (df["confidence"] < 0.5).sum()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.75rem;">
                    <div class="metric-value">{avg_conf*100:.1f}%</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.75rem;">
                    <div class="metric-value">{low_conf}</div>
                    <div class="metric-label">Low Confidence (&lt;50%)</div>
                </div>
                """, unsafe_allow_html=True)

            # Results table
            st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)

            display_df = df[["ticket_text", "predicted_department", "confidence"]].copy()
            display_df.columns = ["Ticket", "Department", "Confidence"]
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

            # Download
            csv_output = df.to_csv(index=False)
            st.download_button(
                label="↓ Download Results CSV",
                data=csv_output,
                file_name=f"classified_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def page_history():
    """Classification history page."""
    st.markdown('<div class="section-header">Classification History</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: var(--text-muted);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            <div>No tickets classified yet. Go to Classify to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary cards
    dept_counts = {}
    total_conf = 0
    for h in history:
        dept_counts[h["department"]] = dept_counts.get(h["department"], 0) + 1
        total_conf += h["confidence"]

    cols = st.columns(len(DEPARTMENT_CONFIG) + 1)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-value">{len(history)}</div>
            <div class="metric-label">Total</div>
        </div>
        """, unsafe_allow_html=True)

    for col, (dept, config) in zip(cols[1:], DEPARTMENT_CONFIG.items()):
        count = dept_counts.get(dept, 0)
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 1.25rem;">{config['icon']}</div>
                <div class="metric-value" style="font-size: 1.25rem; color: {config['color']};">{count}</div>
                <div class="metric-label">{dept.split()[0]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # History list
    for i, entry in enumerate(history[:50]):
        dept = entry["department"]
        color = get_dept_color(dept)
        icon = get_dept_icon(dept)
        conf = entry["confidence"]
        text = entry["text"][:120] + ("..." if len(entry["text"]) > 120 else "")
        ts = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")

        st.markdown(f"""
        <div class="history-row">
            <div class="history-dept-badge" style="background: {color}18; color: {color};">
                {icon} {dept}
            </div>
            <div class="history-text">{text}</div>
            <div class="history-conf" style="color: {color};">{conf*100:.1f}%</div>
            <div class="history-time">{ts}</div>
        </div>
        """, unsafe_allow_html=True)

    if len(history) > 0:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


def page_analytics():
    """Analytics dashboard page."""
    st.markdown('<div class="section-header">Analytics Dashboard</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if len(history) < 2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: var(--text-muted);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div>Classify at least 2 tickets to see analytics.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(history)

    # Department distribution
    st.markdown("**Department Distribution**")
    dept_counts = df["department"].value_counts()
    chart_df = pd.DataFrame({
        "Department": dept_counts.index,
        "Count": dept_counts.values
    }).set_index("Department")
    st.bar_chart(chart_df)

    # Confidence distribution
    st.markdown("**Confidence Distribution**")
    conf_df = pd.DataFrame({
        "Ticket": range(len(df)),
        "Confidence": df["confidence"].values * 100
    }).set_index("Ticket")
    st.line_chart(conf_df)

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-value">{df['confidence'].mean()*100:.1f}%</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-value">{df['confidence'].min()*100:.1f}%</div>
            <div class="metric-label">Min Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-value">{df['confidence'].max()*100:.1f}%</div>
            <div class="metric-label">Max Confidence</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_header()

    # Load model
    with st.spinner("Loading model..."):
        classifier = load_classifier()

    render_sidebar(classifier)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Classify",
        "📦 Batch",
        "📋 History",
        "📊 Analytics",
    ])

    with tab1:
        page_classify(classifier)
    with tab2:
        page_batch(classifier)
    with tab3:
        page_history()
    with tab4:
        page_analytics()


if __name__ == "__main__":
    main()
