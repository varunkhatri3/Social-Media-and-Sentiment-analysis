"""
Local sentiment demo (Streamlit).

From the project root:
    streamlit run app.py

Requires `models/tfidf_vectorizer.pkl` and `models/best_model.pkl`
(run `feature_extraction.ipynb` and `model_training.ipynb` first).
"""

from __future__ import annotations
import streamlit as st
import joblib

import streamlit as st

from predict_sentiment import predict_sentiment
from text_preprocessing import preprocess_text

if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""


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


st.set_page_config(
    page_title="Sentiment analysis",
    page_icon="💬",
    layout="centered",
)

if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""

st.title("Social media sentiment")
st.caption("Multiclass model: negative · neutral · positive (same preprocessing as training).")

with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. Enter tweet-style text below.\n"
        "2. Click **Analyze**.\n"
        "3. For many lines, open **Bulk**."
    )
    st.divider()
    st.markdown("### Prerequisites")
    st.markdown(
        "Trained artifacts must exist:\n"
        "- `models/tfidf_vectorizer.pkl`\n"
        "- `models/best_model.pkl`"
    )
    if st.button("Try a sample"):
        st.session_state["main_input"] = (
            "This update made my day — smooth, fast, and exactly what I needed!"
        )
        st.rerun()

text = st.text_area(
    "Text",
    height=140,
    placeholder="Paste a tweet or short post…",
    key="main_input",
)

c1, c2 = st.columns(2)
analyze = c1.button("Analyze", type="primary")
# AFTER (fixed)
def clear_input():
    st.session_state["main_input"] = ""

st.button("Clear", on_click=clear_input)
   

if analyze:
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        with st.spinner("Loading model / NLTK (first run can take a moment)…"):
            try:
                label = predict_sentiment(text)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()
        _show_result(label)
        with st.expander("Preprocessed text (what the model sees)"):
            cleaned = preprocess_text(text, use_pos_tag=True)
            st.code(cleaned or "(empty after cleaning)", language=None)

with st.expander("Bulk: one sample per line"):
    bulk = st.text_area("Lines", height=160, key="bulk", label_visibility="collapsed")
    if st.button("Analyze all lines", key="bulk_btn"):
        lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
        if not lines:
            st.warning("Add at least one non-empty line.")
        else:
            with st.spinner("Scoring…"):
                try:
                    rows = [{"text": ln, "sentiment": predict_sentiment(ln)} for ln in lines]
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.exception(e)
                    st.stop()
            st.dataframe(rows, use_container_width=True, hide_index=True)
