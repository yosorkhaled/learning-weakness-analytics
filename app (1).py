import streamlit as st
import pdfplumber
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from textblob import TextBlob
from groq import Groq

st.set_page_config(
    page_title="Learning Weakness Analytics",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; color: #f0f0f0; }
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
    .result-box {
        background: #1a237e22;
        border: 2px solid #3f51b5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .result-content {
        color: #e0e0e0;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        line-height: 1.7;
        border-left: 3px solid #3f51b5;
        padding-left: 0.8rem;
    }
    .answer-box {
        background: #0d2b1f;
        border: 2px solid #2e7d52;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .answer-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #4caf90;
        margin-bottom: 0.5rem;
    }
    .answer-text {
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .no-file-warning {
        background: #1a1d27;
        border: 1px dashed #3e4465;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #78909c;
        font-size: 0.9rem;
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
# Technical terms protection
# ─────────────────────────────────────────────
TECHNICAL_TERMS = {
    "NaN", "null", "None", "True", "False", "API", "JSON", "PDF",
    "EDA", "ML", "AI", "KYD", "VOC", "ABA", "SQL", "CSV", "URL",
    "HTTP", "HTTPS", "DataFrame", "numpy", "pandas", "sklearn"
}

def correct_spelling(text):
    masked = text
    placeholders = {}
    for i, term in enumerate(TECHNICAL_TERMS):
        placeholder = f"ZZZTECHZZZ{i}ZZZ"
        placeholders[placeholder] = term
        masked = re.sub(rf'\b{term}\b', placeholder, masked, flags=re.IGNORECASE)
    corrected = str(TextBlob(masked).correct())
    for placeholder, term in placeholders.items():
        corrected = corrected.replace(placeholder, term)
    return corrected

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
                "raw_content": raw_text or "",   # ← للـ embedding والـ LLM
                "content": cleaned,               # ← للعرض في الـ UI فقط
                "word_count": len(cleaned.split()) if cleaned else 0
            })
    return slides

@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

def build_faiss_index(slides):
    model = load_model()
    valid_slides = [s for s in slides if s["raw_content"] or s["content"]]
    # ← استخدم raw_content للـ embedding
    texts = [s["raw_content"] or s["content"] for s in valid_slides]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, valid_slides

def expand_query(question, api_key):
    """Use LLM to rewrite question with technical keywords closer to slide content."""
    try:
        client = Groq(api_key=api_key)
        prompt = f"""Rewrite this student question using technical keywords that would appear in lecture slides.
Return ONLY the rewritten question, nothing else. Keep it short (max 20 words).

Question: {question}
Rewritten:"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
        )
        return response.choices[0].message.content.strip()
    except:
        return question

def retrieve_top_slides(question, valid_slides, faiss_index, api_key, top_k=3):
    """Retrieve top-k most relevant slides using dual embedding + query expansion."""
    model = load_model()

    # Step 1: spell correction
    corrected = correct_spelling(question)

    # Step 2: query expansion
    expanded = expand_query(corrected, api_key)

    # Step 3: encode الاثنين وخذ المتوسط — أدق من الاعتماد على واحد بس
    emb_original = np.array(model.encode([corrected])).astype("float32")
    emb_expanded = np.array(model.encode([expanded])).astype("float32")
    q_emb_np = (emb_original + emb_expanded) / 2
    faiss.normalize_L2(q_emb_np)

    # Step 4: adaptive k based on document size
    n = len(valid_slides)
    if n <= 50:
        k = min(2, n)
    elif n <= 100:
        k = min(3, n)
    else:
        k = min(5, n)

    distances, indices = faiss_index.search(q_emb_np, k=k)
    top_slides = [valid_slides[i] for i in indices[0] if i < len(valid_slides)]

    return top_slides

def get_llm_answer(question, top_slides, api_key):
    """LLM answers based on multiple slides — يستخدم raw_content للجواب الأدق."""
    client = Groq(api_key=api_key)

    # ← عنوان + محتوى كامل بدون تقطيع عشان الـ LLM يشوف السياق الكامل
    slides_context = "\n\n".join([
        f"Slide {s['slide_id']} | Title: {(s['raw_content'] or s['content'])[:100]}\nContent: {(s['raw_content'] or s['content'])}"
        for s in top_slides
    ])

    prompt = f"""You are a helpful teaching assistant. A student asked a question about lecture slides.

Here are the most relevant slides:
{slides_context}

Student question: {question}

Answer the student's question based ONLY on the slide content above.
- Be clear and concise
- If the answer spans multiple slides, mention which slides
- If the answer is not in the slides, say "This topic is not covered in the provided slides."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,   # ← رفعنا من 400 لـ 600
    )
    return response.choices[0].message.content.strip()

def extract_relevant_snippet(question: str, content: str, max_len: int = 400) -> str:
    question_words = set(question.lower().split())
    sentences = re.split(r'(?<=[.!?])\s+', content)
    if not sentences:
        return content[:max_len]
    best_sentence = max(sentences, key=lambda s: len(set(s.lower().split()) & question_words))
    idx = sentences.index(best_sentence)
    start = max(0, idx - 1)
    end = min(len(sentences), idx + 2)
    snippet = " ".join(sentences[start:end])
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")

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
    4. **Ask** any question → get the matching slide + AI answer
    """)
    st.markdown("---")
    st.session_state["groq_api_key"] = "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"
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
                st.session_state["chat_history"] = []
                with st.spinner("Building search index..."):
                    index, valid_slides = build_faiss_index(slides_data)
                    st.session_state["faiss_index"] = index
                    st.session_state["valid_slides"] = valid_slides
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if "slides_data" in st.session_state:
    slides_data = st.session_state["slides_data"]

    st.markdown("---")

    # STEP 2 — Stats
    st.markdown('<div class="step-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Slides", len(slides_data))
    col2.metric("Non-empty Slides", sum(1 for s in slides_data if s["content"]))
    col3.metric("Total Words", sum(s["word_count"] for s in slides_data))

    st.markdown("---")

    # STEP 3 — Preview
    st.markdown('<div class="step-label">Step 3</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Slides Preview")

    show_all = st.toggle("Show all slides", value=False)
    slides_to_show = slides_data if show_all else slides_data[:5]

    cols = st.columns(2)
    for i, slide in enumerate(slides_to_show):
        with cols[i % 2]:
            preview = slide["raw_content"][:180] + ("…" if len(slide["raw_content"]) > 180 else "") if slide["raw_content"] else "(empty)"
            st.markdown(f"""
            <div class="slide-card">
                <div class="slide-number">Slide {slide['slide_id']} &nbsp;·&nbsp; {slide['word_count']} words</div>
                <div class="slide-content">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

    if not show_all:
        st.caption(f"Showing 5 of {len(slides_data)} slides. Toggle to show all.")

    st.markdown("---")

    # STEP 4 — Download JSON
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
# STEP 5 — Chat Interface
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-label">Step 5</div>', unsafe_allow_html=True)
st.markdown("### 💬 Ask a Question")

if "slides_data" not in st.session_state:
    st.markdown("""
    <div class="no-file-warning">
        📂 Upload and process a PDF first (Step 1) to enable chat.
    </div>
    """, unsafe_allow_html=True)
else:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about your slides...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["chat_history"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching slides..."):
                top_slides = retrieve_top_slides(
                    question,
                    st.session_state["valid_slides"],
                    st.session_state["faiss_index"],
                    st.session_state["groq_api_key"],
                )

            # show slide numbers found
            slide_ids = ", ".join([f"Slide {s['slide_id']}" for s in top_slides])
            st.markdown(f"📌 **Searching in: {slide_ids}**")

            # snippet من raw_content للعرض
            snippet = extract_relevant_snippet(question, top_slides[0]["raw_content"] or top_slides[0]["content"])
            st.markdown(f"""
            <div class="result-box">
                <div class="answer-label" style="color:#7986cb;">📖 From the Slide</div>
                <div class="result-content">{snippet}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Generating answer..."):
                answer = get_llm_answer(
                    question,
                    top_slides,
                    st.session_state["groq_api_key"]
                )

            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">🤖 AI Answer</div>
                <div class="answer-text">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

            full_response = f"📌 **{slide_ids}**\n\n📖 {snippet}\n\n🤖 {answer}"
            st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
