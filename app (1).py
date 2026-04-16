import streamlit as st
import pdfplumber
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from textblob import TextBlob

st.set_page_config(
    page_title="Learning Weakness Analytics",
    page_icon="📚",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; color: #f0f0f0; }

    .section-box {
        background: #1a1d27;
        border: 1px solid #2e3248;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .slide-card {
        background: #1e2130;
        border: 1px solid #2e3248;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: border 0.2s;
    }
    .slide-card:hover { border-color: #5c6bc0; }
    .slide-number {
        font-size: 0.75rem;
        color: #7986cb;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .slide-content {
        color: #cfd8dc;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        line-height: 1.6;
    }
    .search-section {
        background: #12151f;
        border-top: 1px solid #2e3248;
        padding: 2rem;
        margin-top: 2rem;
        border-radius: 14px;
    }
    .search-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e8eaf6;
        margin-bottom: 0.3rem;
    }
    .search-sub {
        font-size: 0.85rem;
        color: #78909c;
        margin-bottom: 1.2rem;
    }
    .result-box {
        background: #1a237e22;
        border: 2px solid #3f51b5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .result-slide-num {
        font-size: 1rem;
        font-weight: 700;
        color: #7986cb;
    }
    .result-content {
        color: #e0e0e0;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        line-height: 1.7;
        border-left: 3px solid #3f51b5;
        padding-left: 0.8rem;
    }
    .no-file-warning {
        background: #1a1d27;
        border: 1px dashed #3e4465;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #78909c;
        font-size: 0.9rem;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
        border-radius: 10px !important;
        background: #1e2130 !important;
        border: 1.5px solid #3e4465 !important;
        color: #f0f0f0 !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #5c6bc0 !important;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .step-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #5c6bc0;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[•▪▸►●◆◇→\-–—]+", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_pdf(uploaded_file) -> list:
    slides = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()
            cleaned = clean_text(raw_text) if raw_text else ""
            slides.append({
                "slide_id": page_num,
                "raw_content": raw_text or "",
                "content": cleaned,
                "word_count": len(cleaned.split()) if cleaned else 0
            })
    return slides

@st.cache_resource
def load_model():
    # Using best performing model (all-mpnet-base-v2)
    return SentenceTransformer("all-mpnet-base-v2")

def build_faiss_index(slides):
    model = load_model()
    valid_slides = [s for s in slides if s["content"]]
    texts = [s["content"] for s in valid_slides]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, valid_slides

def extract_relevant_snippet(question: str, content: str, max_len: int = 300) -> str:
    """Return the part of content most relevant to the question."""
    question_words = set(question.lower().split())
    sentences = re.split(r'(?<=[.!?])\s+', content)
    if not sentences:
        return content[:max_len]
    # Score each sentence by overlap with question words
    best_sentence = max(sentences, key=lambda s: len(set(s.lower().split()) & question_words))
    # Return best sentence + surrounding context
    idx = sentences.index(best_sentence)
    start = max(0, idx - 1)
    end = min(len(sentences), idx + 2)
    snippet = " ".join(sentences[start:end])
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")

def search_slide(question):
    if "faiss_index" not in st.session_state:
        return None, None
    model = load_model()
    corrected = str(TextBlob(question).correct())
    q_emb = np.array(model.encode([corrected])).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = st.session_state["faiss_index"].search(q_emb, k=1)
    matched_slide = st.session_state["valid_slides"][indices[0][0]]
    snippet = extract_relevant_snippet(question, matched_slide["content"])
    return corrected, {"slide_id": matched_slide["slide_id"], "snippet": snippet}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📚 Learning Weakness Analytics")
st.markdown("Upload your lecture slides, explore the content, and ask any question to find the right slide instantly.")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ How it works")
    st.markdown("""
    1. **Upload** your PDF slides  
    2. **Explore** each slide's content  
    3. **Download** the cleaned JSON  
    4. **Ask** any question → get the matching slide  
    """)
    st.markdown("---")
    if "slides_data" in st.session_state:
        s = st.session_state["slides_data"]
        st.metric("Slides loaded", len(s))
        st.metric("Non-empty", sum(1 for x in s if x["content"]))

# ─────────────────────────────────────────────
# STEP 1 — Upload
# ─────────────────────────────────────────────
st.markdown('<div class="step-label">Step 1</div>', unsafe_allow_html=True)
st.markdown("### 📤 Upload your PDF slides")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    st.success(f"✅ **{uploaded_file.name}** uploaded successfully")

    if st.button("🚀 Parse & Clean PDF", type="primary"):
        with st.spinner("Processing..."):
            try:
                slides_data = parse_pdf(uploaded_file)
                st.session_state["slides_data"] = slides_data
                st.session_state["filename"] = uploaded_file.name
                # Build index right after parsing
                with st.spinner("Building search index..."):
                    index, valid_slides = build_faiss_index(slides_data)
                    st.session_state["faiss_index"] = index
                    st.session_state["valid_slides"] = valid_slides
                st.session_state.pop("search_result", None)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if "slides_data" in st.session_state:
    slides_data = st.session_state["slides_data"]

    st.markdown("---")

    # ─────────────────────────────────────────────
    # STEP 2 — Stats
    # ─────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Slides", len(slides_data))
    col2.metric("Non-empty Slides", sum(1 for s in slides_data if s["content"]))
    col3.metric("Total Words", sum(s["word_count"] for s in slides_data))

    st.markdown("---")

    # ─────────────────────────────────────────────
    # STEP 3 — Preview slides as cards
    # ─────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 3</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Slides Preview")

    cols = st.columns(2)
    for i, slide in enumerate(slides_data):
        with cols[i % 2]:
            preview = slide["raw_content"][:180] + ("…" if len(slide["raw_content"]) > 180 else "") if slide["raw_content"] else "(empty)"
            st.markdown(f"""
            <div class="slide-card">
                <div class="slide-number">Slide {slide['slide_id']} &nbsp;·&nbsp; {slide['word_count']} words</div>
                <div class="slide-content">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # STEP 4 — Download JSON
    # ─────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 4</div>', unsafe_allow_html=True)
    st.markdown("### 💾 Download Dataset")

    output_json = [{"slide_id": s["slide_id"], "content": s["content"]} for s in slides_data]
    json_str = json.dumps(output_json, ensure_ascii=False, indent=2)
    fname = st.session_state["filename"].replace(".pdf", "")

    col_prev, col_dl = st.columns([2, 1])
    with col_prev:
        st.json(output_json[:2])
        st.caption("Showing first 2 slides only.")
    with col_dl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download JSON File",
            data=json_str.encode("utf-8"),
            file_name=f"{fname}_cleaned.json",
            mime="application/json",
            type="primary"
        )

# ─────────────────────────────────────────────
# STEP 5 — Search bar (always visible at bottom)
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-label">Step 5</div>', unsafe_allow_html=True)
st.markdown("### 🔎 Ask a Question")
st.markdown("Type any question about your slides and we'll find the most relevant slide for you.")

if "slides_data" not in st.session_state:
    st.markdown("""
    <div class="no-file-warning">
        📂 Upload and process a PDF first (Step 1) to enable search.
    </div>
    """, unsafe_allow_html=True)

question = st.text_input(
    label="question",
    placeholder=" Ask anything",
    label_visibility="collapsed",
)

search_clicked = st.button("🔍 Find Slide", type="primary", disabled=("faiss_index" not in st.session_state))

if search_clicked and question.strip():
    with st.spinner("Searching..."):
        corrected, result = search_slide(question)

    if result:
        if corrected.lower() != question.strip().lower():
            st.info(f"✏️ Spell-corrected to: **{corrected}**")

        st.markdown(f"""
        <div class="result-box">
            <div class="result-slide-num">📌 Slide {result['slide_id']}</div>
            <div class="result-content">{result['snippet']}</div>
        </div>
        """, unsafe_allow_html=True)
elif search_clicked and not question.strip():
    st.warning("⚠️ Please enter a question first.")

else:
    st.info("👆 Upload a PDF file to get started.")
