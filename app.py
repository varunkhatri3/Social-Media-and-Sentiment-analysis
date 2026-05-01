from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict_sentiment import predict_sentiment
from text_preprocessing import preprocess_text


SYNTHETIC_REPORT_PATH = Path("data/classification_report.txt")
REALISTIC_REPORT_PATH = Path("data/realistic_classification_report.txt")
SYNTHETIC_CONFUSION_PATH = Path("notebooks/confusion_matrix.png")
REALISTIC_CONFUSION_PATH = Path("data/realistic_confusion_matrix.png")


if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-accent: linear-gradient(145deg, #07111f 0%, #0b1729 42%, #13233d 100%);
            --panel-bg: rgba(9, 18, 34, 0.78);
            --panel-border: rgba(101, 173, 255, 0.16);
            --text-main: #f4f7fb;
            --text-soft: #9fb0c9;
            --primary: #4da3ff;
            --primary-strong: #2f7fe0;
            --shadow: 0 24px 50px rgba(0, 0, 0, 0.34);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(77, 163, 255, 0.2), transparent 24%),
                radial-gradient(circle at 85% 20%, rgba(54, 208, 164, 0.16), transparent 18%),
                radial-gradient(circle at bottom right, rgba(30, 62, 114, 0.35), transparent 28%),
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
            background: rgba(8, 15, 28, 0.92);
            color: var(--text-main);
            border-radius: 18px;
        }

        div[data-baseweb="base-input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(8, 15, 28, 0.92) !important;
            border-color: rgba(77, 163, 255, 0.24) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="textarea"] > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px rgba(77, 163, 255, 0.2) !important;
        }

        .stButton > button {
            border-radius: 999px;
            font-weight: 700;
            padding: 0.65rem 1.2rem;
            background: rgba(14, 25, 44, 0.9);
            color: var(--text-main);
            border: 1px solid rgba(77, 163, 255, 0.18);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.28);
            border-color: rgba(77, 163, 255, 0.34);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #58adff 0%, #2f7fe0 100%);
            color: white;
            border: none;
        }

        [data-testid="stSidebar"] {
            background: rgba(6, 13, 25, 0.88);
            border-right: 1px solid rgba(77, 163, 255, 0.12);
        }

        [data-testid="stExpander"] {
            background: rgba(10, 18, 32, 0.84);
            border: 1px solid rgba(77, 163, 255, 0.12);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stMarkdownContainer"],
        .stCaption,
        label,
        p,
        li,
        span {
            color: var(--text-main);
        }

        [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        [data-testid="stCodeBlock"] {
            border-radius: 16px;
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


@st.cache_data(show_spinner=False)
def _load_classification_report(report_path: str) -> tuple[pd.DataFrame, dict[str, float], int]:
    report_file = Path(report_path)
    if not report_file.exists():
        raise FileNotFoundError(f"Missing evaluation report: {report_file}")

    lines = [line.strip() for line in report_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_rows: list[dict[str, float | str]] = []
    summary: dict[str, float] = {}
    total_support = 0

    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 5:
            label, precision, recall, f1_score, support = parts
            class_rows.append(
                {
                    "label": label,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                    "support": int(support),
                }
            )
        elif parts and parts[0] == "accuracy" and len(parts) >= 3:
            summary["accuracy"] = float(parts[-2])
            total_support = int(parts[-1])
        elif len(parts) >= 5 and "avg" in parts:
            key = "_".join(parts[:-4])
            summary[f"{key}_precision"] = float(parts[-4])
            summary[f"{key}_recall"] = float(parts[-3])
            summary[f"{key}_f1_score"] = float(parts[-2])
            summary[f"{key}_support"] = int(parts[-1])

    if not class_rows:
        raise ValueError("Classification report could not be parsed.")

    return pd.DataFrame(class_rows), summary, total_support


def _render_evaluation_panel(
    title: str,
    report_path: Path,
    confusion_path: Path | None,
    source_note: str,
    show_perfect_score_warning: bool = False,
) -> None:
    metrics_df, summary, total_support = _load_classification_report(str(report_path))

    st.markdown(f"#### {title}")
    st.caption(source_note)

    accuracy = summary.get("accuracy", 0.0)
    macro_precision = summary.get("macro_avg_precision", 0.0)
    macro_recall = summary.get("macro_avg_recall", 0.0)
    macro_f1 = summary.get("macro_avg_f1_score", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{macro_precision:.2%}")
    c3.metric("Recall", f"{macro_recall:.2%}")
    c4.metric("F1 Score", f"{macro_f1:.2%}")

    if (
        show_perfect_score_warning
        and accuracy >= 0.999
        and macro_precision >= 0.999
        and macro_recall >= 0.999
        and macro_f1 >= 0.999
    ):
        st.warning(
            "These near-perfect metrics are likely inflated by the synthetic, repetitive dataset. "
            "They are useful for demonstrating the pipeline, but they should not be treated as realistic "
            "real-world sentiment performance."
        )
        st.caption(
            "The train/test split is already stratified and the TF-IDF vectorizer is fit on training data only. "
            "The bigger limitation here is dataset simplicity, not an obvious evaluation bug."
        )

    chart_df = (
        metrics_df[["label", "precision", "recall", "f1_score"]]
        .set_index("label")
        .rename(columns={"f1_score": "f1 score"})
    )
    st.bar_chart(chart_df, use_container_width=True)

    pie_left, pie_mid, pie_right = st.columns([1, 1, 1.15])

    with pie_left:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.pie(
            metrics_df["support"],
            labels=metrics_df["label"].str.title(),
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax.set_title("Class Distribution")
        ax.axis("equal")
        st.pyplot(fig, clear_figure=True)

    with pie_mid:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        correct = round(accuracy * total_support)
        incorrect = max(total_support - correct, 0)
        values = [correct, incorrect]
        labels = ["Correct", "Incorrect"]
        if incorrect == 0:
            values = [correct]
            labels = ["Correct"]
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4da3ff", "#ff6b6b"][: len(values)],
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1},
        )
        ax.set_title("Accuracy Split")
        ax.axis("equal")
        st.pyplot(fig, clear_figure=True)

    with pie_right:
        if confusion_path and confusion_path.exists():
            st.image(str(confusion_path), caption="Confusion matrix", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with st.expander("Evaluation table"):
        display_df = metrics_df.copy()
        for col in ["precision", "recall", "f1_score"]:
            display_df[col] = display_df[col].map(lambda value: f"{value:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _show_evaluation_dashboard() -> None:
    st.markdown("### Model evaluation")

    panels: list[tuple[str, Path, Path | None, str, bool]] = []
    if REALISTIC_REPORT_PATH.exists():
        panels.append(
            (
                "Realistic benchmark",
                REALISTIC_REPORT_PATH,
                REALISTIC_CONFUSION_PATH,
                "Measured on a curated set of more natural social-media-style sentences. "
                "This is the better indicator of real-world performance.",
                False,
            )
        )
    if SYNTHETIC_REPORT_PATH.exists():
        panels.append(
            (
                "Synthetic holdout",
                SYNTHETIC_REPORT_PATH,
                SYNTHETIC_CONFUSION_PATH,
                "Measured on the synthetic project dataset. Useful for pipeline validation, but easier than real data.",
                True,
            )
        )

    if not panels:
        st.info("No evaluation artifacts were found.")
        return

    if len(panels) == 1:
        title, report_path, confusion_path, source_note, warn = panels[0]
        _render_evaluation_panel(title, report_path, confusion_path, source_note, warn)
        return

    tab_labels = [panel[0] for panel in panels]
    tabs = st.tabs(tab_labels)
    for tab, panel in zip(tabs, panels):
        title, report_path, confusion_path, source_note, warn = panel
        with tab:
            _render_evaluation_panel(title, report_path, confusion_path, source_note, warn)


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
        <h1 class="hero-title">Feedback  Sentiment Analysis</h1>
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
        _show_evaluation_dashboard()
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
