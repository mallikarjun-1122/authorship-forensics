import pandas as pd
import streamlit as st

from backend.ai_detector import detect_ai_text
from backend.authorship import verify_author
from backend.drift import detect_style_drift
from backend.preprocess import clean_text
from backend.report_generator import create_report
from backend.stylometry import feature_names

MIN_WORDS = 20
MAX_FILE_BYTES = 5 * 1024 * 1024

st.set_page_config(
    page_title="Authorship Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Refined Black, White, and Light Green
st.markdown(
    """
    <style>
    /* Global Base */
    .stApp {
        background-color: #0b0d0c;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    p, label {
        color: #d1d5db !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111413 !important;
        border-right: 1px solid #1c2420 !important;
    }
    
    /* Inputs */
    .stTextArea textarea {
        background-color: #141816 !important;
        color: #ffffff !important;
        border: 1px solid #232e28 !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #34d399 !important;
        box-shadow: 0 0 0 1px #34d399 !important;
    }
    
    /* Buttons */
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        background-color: #10b981 !important;
        color: #051a10 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        background-color: #34d399 !important;
        color: #000000 !important;
        box-shadow: 0 0 15px rgba(52, 211, 153, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Metric Cards */
    .score-card {
        background-color: #111413;
        border: 1px solid #1c2420;
        border-top: 3px solid #10b981;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-title {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .score-val {
        color: #34d399;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .score-sub {
        color: #e5e7eb;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }
    
    /* Content Panels */
    .panel-box {
        background-color: #111413;
        border: 1px solid #1c2420;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .panel-header {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .panel-header-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Tags / Badges */
    .chip {
        display: inline-block;
        background-color: #17211b;
        color: #6ee7b7;
        border: 1px solid #243d2f;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.82rem;
        font-family: monospace;
    }
    .chip-muted {
        display: inline-block;
        background-color: #141715;
        color: #9ca3af;
        border: 1px solid #232a26;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.82rem;
        font-family: monospace;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pill-green {
        background-color: #172e22;
        color: #34d399;
        border: 1px solid #10b981;
    }
    .pill-white {
        background-color: #1e2421;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    /* Header Bar */
    .top-header {
        border-bottom: 1px solid #1c2420;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .top-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .top-desc {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Bar
st.markdown(
    """
    <div class="top-header">
        <div class="top-title">🔍 Authorship Forensics & AI Detector</div>
        <div class="top-desc">Quantitative Stylometric Fingerprinting • Synthetic Text Detection • Intra-Doc Drift</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def load_sample_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


# Sidebar: Input Mode & Execution
st.sidebar.markdown("<h3 style='color: #ffffff; font-size: 1.1rem; margin-bottom: 0.2rem;'>Input Source</h3>", unsafe_allow_html=True)
input_mode = st.sidebar.radio(
    "Select Input Method",
    ["⚡ Quick Demo (1-Click)", "✍️ Paste Text", "📁 Upload .txt Files"],
    label_visibility="collapsed",
)

known_texts = []
unknown_text = ""
target_label = "Questioned Document"

if input_mode == "⚡ Quick Demo (1-Click)":
    demo_choice = st.sidebar.selectbox(
        "Choose Demo Preset",
        [
            "Match Demo: Known Author vs Same Author",
            "Mismatch Demo: Known Author vs AI-Generated Text",
        ],
    )
    if demo_choice == "Match Demo: Known Author vs Same Author":
        k1 = load_sample_file("backend/sample_data/author1.txt")
        uk = load_sample_file("backend/sample_data/unknown.txt")
        known_texts = [k1]
        unknown_text = uk
        target_label = "unknown.txt (Human Sample)"
    else:
        k1 = load_sample_file("backend/sample_data/author1.txt")
        ai = load_sample_file("backend/sample_data/ai_doc.txt")
        known_texts = [k1]
        unknown_text = ai
        target_label = "ai_doc.txt (Synthetic Sample)"

    st.sidebar.info("💡 Preset loaded. Click **'Run Analysis'** below to evaluate.")

elif input_mode == "✍️ Paste Text":
    raw_known = st.sidebar.text_area(
        "Known Author Sample(s)",
        placeholder="Paste known verified writing here (minimum 20 words)...",
        height=140,
    )
    raw_unknown = st.sidebar.text_area(
        "Document to Verify",
        placeholder="Paste target document to analyze here (minimum 20 words)...",
        height=140,
    )
    if raw_known.strip():
        known_texts = [clean_text(raw_known)]
    if raw_unknown.strip():
        unknown_text = clean_text(raw_unknown)

elif input_mode == "📁 Upload .txt Files":
    uploaded_known = st.sidebar.file_uploader(
        "Known Author References",
        type=["txt"],
        accept_multiple_files=True,
    )
    uploaded_unknown = st.sidebar.file_uploader(
        "Questioned Document",
        type=["txt"],
    )
    if uploaded_known:
        for f in uploaded_known:
            content = f.getvalue().decode("utf-8-sig", errors="ignore").strip()
            if content:
                known_texts.append(clean_text(content))
    if uploaded_unknown:
        unknown_text = clean_text(uploaded_unknown.getvalue().decode("utf-8-sig", errors="ignore").strip())
        target_label = uploaded_unknown.name

# Action Buttons
st.sidebar.markdown("<br>", unsafe_allow_html=True)
col_a1, col_a2 = st.sidebar.columns([1.2, 1])
with col_a1:
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
with col_a2:
    reset_btn = st.button("Reset", use_container_width=True)

# Project Documentation Download Button in Sidebar
try:
    with open("Authorship_Forensics_Project_Documentation.docx", "rb") as f_doc:
        doc_bytes = f_doc.read()
    st.sidebar.markdown("<hr style='border-color: #1c2420;'>", unsafe_allow_html=True)
    st.sidebar.download_button(
        label="📄 Download Project Report (.docx)",
        data=doc_bytes,
        file_name="Authorship_Forensics_Project_Documentation.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
except Exception:
    pass

if reset_btn:
    if "results" in st.session_state:
        del st.session_state["results"]
    st.rerun()

# Execution Handler
if run_btn:
    if not known_texts or not unknown_text:
        st.sidebar.error("Provide both known author sample(s) and the target document.")
    else:
        # Validate word counts
        k_short = [i + 1 for i, t in enumerate(known_texts) if len(t.split()) < MIN_WORDS]
        u_words = len(unknown_text.split())

        if u_words < MIN_WORDS:
            st.sidebar.error(f"Target document has only {u_words} words (minimum {MIN_WORDS} required).")
        elif k_short:
            st.sidebar.error(f"Known sample #{k_short[0]} has fewer than {MIN_WORDS} words.")
        else:
            with st.spinner("Analyzing stylometric fingerprint..."):
                auth_res = verify_author(known_texts, unknown_text)
                ai_res = detect_ai_text(unknown_text)
                drift_res = detect_style_drift(unknown_text)

                st.session_state["results"] = {
                    "authorship": auth_res,
                    "ai": ai_res,
                    "drift": drift_res,
                    "label": target_label,
                    "unknown_words": u_words,
                    "known_count": len(known_texts),
                }

# Main Area Display
if "results" not in st.session_state:
    st.markdown(
        """
        <div class="panel-box">
            <div class="panel-header">
                <span class="panel-header-dot"></span>
                Getting Started
            </div>
            <p style="margin: 0; color: #d1d5db;">
                Select an input option in the sidebar:
            </p>
            <ul style="color: #9ca3af; margin-top: 0.5rem; line-height: 1.6;">
                <li><strong style="color: #ffffff;">⚡ Quick Demo:</strong> Run an instant test with pre-packaged reference data.</li>
                <li><strong style="color: #ffffff;">✍️ Paste Text:</strong> Type or paste custom articles directly.</li>
                <li><strong style="color: #ffffff;">📁 Upload Files:</strong> Attach text files (.txt) for batch verification.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Direct button on the main page too
    try:
        with open("Authorship_Forensics_Project_Documentation.docx", "rb") as f_doc:
            doc_data = f_doc.read()
        st.download_button(
            label="📥 Download Complete Project Documentation (.docx)",
            data=doc_data,
            file_name="Authorship_Forensics_Project_Documentation.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception:
        pass

else:
    res = st.session_state["results"]
    auth = res["authorship"]
    ai = res["ai"]
    drift = res["drift"]

    # 1. Top Executive KPI Cards
    col_m1, col_m2, col_m3 = st.columns(3)

    adv_score = auth["advanced_score"]
    if adv_score >= 65:
        match_desc = "Strong Style Match"
        match_pill = "pill-green"
    elif adv_score >= 40:
        match_desc = "Moderate Similarity"
        match_pill = "pill-white"
    else:
        match_desc = "Low Similarity (Unlikely Author)"
        match_pill = "pill-white"

    ai_score = ai["score"]
    if ai_score >= 60:
        ai_desc = "High Synthetic Probability"
        ai_pill = "pill-white"
    elif ai_score >= 30:
        ai_desc = "Moderate AI Markers"
        ai_pill = "pill-white"
    else:
        ai_desc = "Low AI Risk (Human Characteristics)"
        ai_pill = "pill-green"

    drift_flags = [p.get("flag", "Normal") for p in drift]
    has_drift = any("High" in f for f in drift_flags)
    if has_drift:
        drift_desc = "Stylistic Shifts Detected"
        drift_pill = "pill-white"
    elif len(drift) > 1:
        drift_desc = "Consistent Flow Across Sections"
        drift_pill = "pill-green"
    else:
        drift_desc = "Single Paragraph Document"
        drift_pill = "pill-white"

    with col_m1:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">Authorship Match</div>
                <div class="score-val">{adv_score}%</div>
                <div class="score-sub"><span class="status-pill {match_pill}">{match_desc}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">Synthetic AI Risk</div>
                <div class="score-val">{ai_score} <span style="font-size: 1rem; color: #9ca3af;">/ 100</span></div>
                <div class="score-sub"><span class="status-pill {ai_pill}">{ai_desc}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">Paragraph Style Stability</div>
                <div class="score-val">{len(drift)} <span style="font-size: 1rem; color: #9ca3af;">Sections</span></div>
                <div class="score-sub"><span class="status-pill {drift_pill}">{drift_desc}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Detailed Forensic Insights (Clean 2-Column Layout)
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        # Authorship & Stylometry Panel
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-header">
                    <span class="panel-header-dot"></span>
                    Authorship Fingerprint & Explainable Traits
                </div>
            """,
            unsafe_allow_html=True,
        )

        xai = auth.get("xai", {})
        shared = xai.get("shared", [])
        missing = xai.get("missing", [])

        st.caption("Dominant Stylistic Traits Shared with Known Author:")
        if shared:
            chips = "".join(f"<span class='chip'>{w}</span>" for w in shared[:12])
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>No strong recurring n-gram overlap found.</p>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.caption("Known Author Habits Absent in Questioned Document:")
        if missing:
            chips_m = "".join(f"<span class='chip-muted'>{w}</span>" for w in missing[:8])
            st.markdown(f"<div>{chips_m}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>No major absent traits detected.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Style Drift Panel
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-header">
                    <span class="panel-header-dot"></span>
                    Intra-Document Paragraph Consistency
                </div>
            """,
            unsafe_allow_html=True,
        )
        if drift:
            df_drift = pd.DataFrame(drift)
            st.dataframe(
                df_drift.rename(columns={
                    "paragraph": "Section",
                    "similarity": "Consistency (%)",
                    "words": "Word Count",
                    "flag": "Forensic Status",
                }),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No multi-paragraph segments found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # AI Detection Panel
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-header">
                    <span class="panel-header-dot"></span>
                    Synthetic / LLM Discourse Indicators
                </div>
            """,
            unsafe_allow_html=True,
        )

        indicators = ai.get("indicators", [])
        hits = ai.get("matched_count", len(indicators))
        burstiness = ai.get("burstiness", 0.0)
        penalty = ai.get("burst_penalty", 0)

        st.caption(f"Detected Generic AI Discourse Markers ({hits} occurrences):")
        if indicators:
            ai_chips = "".join(f"<span class='chip'>{ind}</span>" for ind in indicators)
            st.markdown(f"<div>{ai_chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #6ee7b7; font-size: 0.85rem;'>✓ Zero cliché AI transition words detected.</p>", unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.caption("Sentence Rhythm & Variation (Burstiness):")
        if penalty > 0:
            st.markdown(
                f"<p style='color: #ffffff; font-size: 0.9rem;'>Unusually uniform sentence length detected "
                f"(Variance score: <strong style='color: #34d399;'>{burstiness}</strong>). "
                "Synthetic text commonly generates monotone cadence.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color: #ffffff; font-size: 0.9rem;'>Natural sentence length variation detected "
                f"(Variance score: <strong style='color: #34d399;'>{burstiness}</strong>). "
                "Human writing typically demonstrates dynamic syntactic shifts.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature Vector Comparison Chart
        st.markdown(
            """
            <div class="panel-box">
                <div class="panel-header">
                    <span class="panel-header-dot"></span>
                    Feature Profile Alignment
                </div>
            """,
            unsafe_allow_html=True,
        )
        features = feature_names()
        df_feat = pd.DataFrame({
            "Feature": features,
            "Known Author": auth["profile_vector"],
            "Questioned": auth["unknown_vector"],
        }).set_index("Feature")

        st.bar_chart(
            df_feat,
            color=["#34d399", "#ffffff"],
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Action Section: Download PDF
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    pdf_data = create_report(auth, ai, drift)
    st.download_button(
        label="📥 Download Forensic PDF Report",
        data=pdf_data,
        file_name=f"Forensic_Report_{res['label']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
