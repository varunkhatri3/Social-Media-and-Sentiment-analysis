"""
Local sentiment demo (Streamlit).

From the project root:
    streamlit run app.py

Requires `models/tfidf_vectorizer.pkl` and `models/best_model.pkl`
(run `feature_extraction.ipynb` and `model_training.ipynb` first).
"""

from __future__ import annotations

import streamlit as st

from predict_sentiment import predict_sentiment
from text_preprocessing import preprocess_text


if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-accent: linear-gradient(135deg, #fff4e8 0%, #ffe4d4 45%, #ffd7c2 100%);
            --panel-bg: rgba(255, 248, 242, 0.78);
            --panel-border: rgba(184, 91, 59, 0.18);
            --text-main: #2d1f1a;
            --text-soft: #6d554b;
            --primary: #c4572e;
            --shadow: 0 18px 40px rgba(112, 54, 35, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.92), transparent 32%),
                radial-gradient(circle at bottom right, rgba(255, 214, 186, 0.58), transparent 26%),
                var(--bg-accent);
        }

        .block-container {
            max-width: 900px;
            padding-top: 2.4rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: var(--panel-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.4rem;
            box-shadow: var(--shadow);
        }

        .hero-eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: var(--text-main);
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 0.98;
            font-weight: 800;
            margin: 0;
        }

        .hero-copy {
            color: var(--text-soft);
            font-size: 1rem;
            margin: 0.8rem 0 0;
        }

        div[data-baseweb="textarea"] textarea {
            background: rgba(255, 250, 246, 0.94);
            color: var(--text-main);
            border-radius: 18px;
        }

        div[data-baseweb="base-input"] > div,
        div[data-baseweb="textarea"] > div {
            border-color: rgba(196, 87, 46, 0.28) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="textarea"] > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px rgba(196, 87, 46, 0.2) !important;
        }

        .stButton > button {
            border-radius: 999px;
            font-weight: 700;
            padding: 0.65rem 1.2rem;
            border: 1px solid rgba(196, 87, 46, 0.18);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(112, 54, 35, 0.12);
            border-color: rgba(196, 87, 46, 0.34);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #db683d 0%, #c4572e 100%);
            color: white;
            border: none;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 246, 239, 0.72);
            border-right: 1px solid rgba(184, 91, 59, 0.16);
        }

        [data-testid="stExpander"] {
            background: rgba(255, 250, 246, 0.78);
            border: 1px solid rgba(184, 91, 59, 0.14);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_result(label: str) -> None:
    label_l = label.lower()
    if label_l == "positive":
        st.success("**Positive**")
    elif label_l == "negative":
        st.error("**Negative**")
    elif label_l == "neutral":
        st.warning("**Neutral**")
    else:
        st.info(f"**{label.title()}**")


def clear_input() -> None:
    st.session_state["main_input"] = ""


st.set_page_config(
    page_title="Sentiment analysis",
    page_icon="💬",
    layout="centered",
)

_inject_styles()

st.markdown(
    """
    <section class="hero-card">
        <div class="hero-eyebrow">Sentiment Intelligence</div>
        <h1 class="hero-title">Social media sentiment</h1>
        <p class="hero-copy">
            Analyze short-form text as negative, neutral, or positive using the
            trained classifier and the same preprocessing pipeline used during training.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. Enter tweet-style text below.\n"
        "2. Click **Analyze**.\n"
        "3. For many lines, open **Bulk**."
    )
    st.divider()
    st.markdown("### Model Artifacts")
    st.markdown(
        "- `models/tfidf_vectorizer.pkl`\n"
        "- `models/best_model.pkl`\n"
        "- `models/vader_lexicon.txt`"
    )
    if st.button("Try a sample"):
        st.session_state["main_input"] = (
            "This update made my day - smooth, fast, and exactly what I needed!"
        )
        st.rerun()

text = st.text_area(
    "Text",
    height=160,
    placeholder="Paste a tweet or short post...",
    key="main_input",
)

c1, c2 = st.columns([1, 1])
analyze = c1.button("Analyze", type="primary", use_container_width=True)
c2.button("Clear", on_click=clear_input, use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        with st.spinner("Loading model and preprocessing resources..."):
            try:
                label = predict_sentiment(text)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()
        _show_result(label)
        with st.expander("Preprocessed text"):
            cleaned = preprocess_text(text, use_pos_tag=True)
            st.code(cleaned or "(empty after cleaning)", language=None)

with st.expander("Bulk analysis: one sample per line"):
    bulk = st.text_area("Lines", height=160, key="bulk", label_visibility="collapsed")
    if st.button("Analyze all lines", key="bulk_btn", use_container_width=True):
        lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
        if not lines:
            st.warning("Add at least one non-empty line.")
        else:
            with st.spinner("Scoring samples..."):
                try:
                    rows = [{"text": ln, "sentiment": predict_sentiment(ln)} for ln in lines]
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.exception(e)
                    st.stop()
            st.dataframe(rows, use_container_width=True, hide_index=True)
