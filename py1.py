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

st.title("📚 Learning Weakness Analytics")
st.markdown("### PDF Parsing & Data Pipeline")
st.markdown("---")

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove bullet symbols
    text = re.sub(r"[•▪▸►●◆◇→\-–—]+", " ", text)
    # 3. Remove noise characters (keeps Arabic + English + numbers)
    text = re.sub(r"[^\w\s\u0600-\u06FF.,!?]", " ", text)
    # 4. Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

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

# ── NEW functions for Step 7 ──────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)

def build_faiss_index(slides, model):
    valid_slides = [s for s in slides if s["content"]]
    texts = [s["content"] for s in valid_slides]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, valid_slides

def search_slide(question, index, model, valid_slides, top_k=3):
    corrected = str(TextBlob(question).correct())
    q_emb = np.array(model.encode([corrected])).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k=top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        results.append({"rank": rank + 1, "slide": valid_slides[idx], "score": float(score)})
    return corrected, results
# ── END NEW functions ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This tool processes your lecture slides:
    1. **Upload** a PDF file
    2. **Parse** each slide/page
    3. **Clean** the text
    4. **Download** as structured JSON
    """)
    st.markdown(" Cleaned structured dataset")

st.markdown("### 📤 Step 1: Upload your PDF slides")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")
    st.markdown("---")
    st.markdown("### ⚙️ Step 2: Process the PDF")

    if st.button("🚀 Parse & Clean PDF", type="primary"):
        with st.spinner("Processing your PDF... please wait"):
            try:
                slides_data = parse_pdf(uploaded_file)
                st.session_state["slides_data"] = slides_data
                st.session_state["filename"] = uploaded_file.name
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if "slides_data" in st.session_state:
        slides_data = st.session_state["slides_data"]
        st.markdown("---")
        st.markdown("### 📊 Step 3: Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Slides", len(slides_data))
        col2.metric("Non-empty Slides", sum(1 for s in slides_data if s["content"]))
        col3.metric("Total Words", sum(s["word_count"] for s in slides_data))

        st.markdown("---")
        st.markdown("### 🔍 Step 4: Preview Slides")

        selected = st.selectbox(
            "Select a slide:",
            options=[f"Slide {s['slide_id']}" for s in slides_data]
        )
        idx = int(selected.split(" ")[1]) - 1
        slide = slides_data[idx]

        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.markdown("**📄 Raw Text**")
            st.text_area("", value=slide["raw_content"] or "(empty)", height=250, key=f"raw_{slide['slide_id']}")
        with col_clean:
            st.markdown("**✨ Cleaned Text**")
            st.text_area("", value=slide["content"] or "(empty)", height=250, key=f"clean_{slide['slide_id']}")

        st.markdown(f"**Word count:** {slide['word_count']} words")
        st.markdown("---")
        st.markdown("### 📋 Step 5: JSON Preview")

        output_json = [{"slide_id": s["slide_id"], "content": s["content"]} for s in slides_data]
        st.json(output_json[:3])
        st.caption("Showing first 3 slides only.")

        st.markdown("---")
        st.markdown("### 💾 Step 6: Download Dataset")

        json_str = json.dumps(output_json, ensure_ascii=False, indent=2)
        fname = st.session_state["filename"].replace(".pdf", "")

        st.download_button(
            label="⬇️ Download JSON File",
            data=json_str.encode("utf-8"),
            file_name=f"{fname}_cleaned.json",
            mime="application/json",
            type="primary"
        )

        # ── NEW: Step 7 ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Step 7: Ask a Question — Find the Right Slide")
        st.caption("Type any question and the AI will find which slide contains the answer.")

        model_choice = st.selectbox(
            "🧠 Choose Embedding Model:",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ],
            help="all-MiniLM-L6-v2 is fastest. all-mpnet-base-v2 is most accurate. Multilingual supports Arabic+English."
        )

        if st.button("⚡ Build Search Index"):
            with st.spinner(f"Loading '{model_choice}' and building index..."):
                model = load_model(model_choice)
                index, valid_slides = build_faiss_index(slides_data, model)
                st.session_state["faiss_index"] = index
                st.session_state["valid_slides"] = valid_slides
                st.session_state["qa_model"] = model
                st.session_state["qa_model_name"] = model_choice
            st.success(f"✅ Index built with {len(valid_slides)} slides using **{model_choice}**")

        if "faiss_index" in st.session_state:
            question = st.text_input(
                "❓ Enter your question:",
                placeholder="e.g. What is machine learning? / what are the data soureces?"
            )

            if st.button("🔍 Find Slide", type="primary"):
                if not question.strip():
                    st.warning("⚠️ Please enter a question first.")
                else:
                    with st.spinner("Searching..."):
                        corrected, results = search_slide(
                            question,
                            st.session_state["faiss_index"],
                            st.session_state["qa_model"],
                            st.session_state["valid_slides"],
                        )

                    if corrected.lower() != question.strip().lower():
                        st.info(f"✏️ Spell-corrected to: **{corrected}**")

                    st.markdown("#### 🎯 Top Results:")
                    medals = ["🥇", "🥈", "🥉"]
                    for r in results:
                        slide_id = r["slide"]["slide_id"]
                        preview = r["slide"]["content"][:300] + ("…" if len(r["slide"]["content"]) > 300 else "")
                        score = round(r["score"] * 100, 1)
                        medal = medals[r["rank"] - 1]
                        if r["rank"] == 1:
                            st.success(f"{medal} **Best Match — Slide {slide_id}** (similarity: {score}%)\n\n{preview}")
                        else:
                            st.info(f"{medal} **#{r['rank']} Match — Slide {slide_id}** (similarity: {score}%)\n\n{preview}")
        # ── END NEW ───────────────────────────────────────────────

else:
    st.info("👆 Upload a PDF file to get started.")
    st.markdown("### 📋 Example Output Format")
    st.json([
        {"slide_id": 1, "content": "introduction to machine learning"},
        {"slide_id": 2, "content": "supervised learning classification regression"},
    ])
